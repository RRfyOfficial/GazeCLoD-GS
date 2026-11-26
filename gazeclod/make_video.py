"""
Session 7 demo video: replay one real NTHU viewer's 60 s walk through the
splat, plain CLoD-GS and GazeCLoD-GS side by side at the SAME number of
Gaussians in view. The red circle is where this viewer was actually looking
(their own eye tracking); the inset zooms on it. The heat comes from the
other team, so the viewer never contributed to it.

Run from the repo root with the clodgs env active, e.g.:
    python gazeclod/make_video.py -m output/truck_clod --scene truck --user user12 --s 16 --heat output/truck_clod/gaze_heat/heat_nthu_truck_mass_sig2.5.npz+output/truck_clod/gaze_heat/heat_nthu_truck_mass_head_sig2.5.npz
Writes <model>/gaze_lod/video_<user>_s<s>_k<k>.mp4 (MPEG-4, plays in VLC and most players)
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import angle_map, focal, make_minicam, project_dir  # noqa: E402
from gazeclod.common import load_model, lod_pipe, render_rgb, to_uint8  # noqa: E402
from gazeclod.eval_gaze_lod import HeatBank  # noqa: E402
from gazeclod.lod_policy import LodState, match_budget, s_vector, user_number  # noqa: E402


def fovea_psnr(img, ref, mask):
    if mask.sum() < 50:
        return float("nan")
    err = ((img - ref) ** 2).mean(0)[torch.as_tensor(mask, device=img.device)]
    return 10 * math.log10(1.0 / max(float(err.mean()), 1e-10))


def panel(img_chw, title, sub, uv, ring, zoom, scale=3):
    im = Image.fromarray(to_uint8(img_chw))
    W = im.width
    d = ImageDraw.Draw(im)
    if uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < W:
        u, v = uv
        x0 = int(min(max(round(u) - zoom // 2, 0), W - zoom))
        y0 = int(min(max(round(v) - zoom // 2, 0), W - zoom))
        inset = im.crop((x0, y0, x0 + zoom, y0 + zoom)).resize((zoom * scale, zoom * scale), Image.NEAREST)
        d.ellipse([u - ring, v - ring, u + ring, v + ring], outline=(255, 40, 40), width=3)
        ix = 8 if u > W / 2 else W - inset.width - 8   # corner opposite the gaze
        iy = 50 if v > W / 2 else W - inset.height - 8
        im.paste(inset, (ix, iy))
        d.rectangle([ix - 2, iy - 2, ix + inset.width + 1, iy + inset.height + 1], outline=(255, 40, 40), width=2)
    d.rectangle([0, 0, W, 44], fill=(0, 0, 0))
    d.text((10, 6), title, fill=(255, 255, 255))
    d.text((10, 24), sub, fill=(200, 200, 200))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", required=True, help="heat .npz, 'a.npz+b.npz' sums files")
    ap.add_argument("--user", required=True, help="test viewer, e.g. user12 (heat comes from the other team)")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--s", type=float, default=16.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=0, help="limit length (0 = whole session)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fovea", type=float, default=5.0)
    ap.add_argument("--zoom", type=int, default=72)
    args = ap.parse_args()

    bank = HeatBank(args.heat)
    parity = user_number(args.user) % 2
    heat_team = [u for u in bank.users if user_number(u) % 2 != parity]
    h = bank.h(heat_team, args.gamma)
    print(f"Viewer {args.user}. Heat from the other team: {', '.join(heat_team)}")

    gs, _ = load_model(args.model, args.iteration)
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    W, fov = args.size, math.radians(args.fov)
    ring = focal(fov, W) * math.tan(math.radians(args.fovea))
    path = {G.user_id(p): p for p in G.list_traces(args.eyenav, args.site, args.scene)}[args.user]
    tr = G.to_scene(G.load_trace(path))
    left = np.where(tr["view"] == 0)[0]
    t = tr["t_ms"][left] / 1000.0
    end = t[-1] if not args.seconds else min(t[-1], t[0] + args.seconds)
    clock = np.arange(t[0], end, 1.0 / args.fps)
    rows = left[np.clip(np.searchsorted(t, clock), 0, len(left) - 1)]

    out = os.path.join(args.model, "gaze_lod", f"video_{args.user}_s{args.s:g}_k{args.k:g}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    footer = 34
    size = (2 * W + 6, W + footer)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    gains = []
    for i in tqdm(rows, desc="frames"):
        R, C, d = tr["R_c2w"][i], tr["C"][i], tr["gaze_d"][i]
        cam = make_minicam(R, C, W, W, fov, fov)
        st = LodState(gs, cam)
        target = st.count(args.s)
        sb, n = match_budget(st, h, args.k, target, args.s)
        with torch.no_grad():
            ref = render_rgb(cam, gs, pipe, bg, s=1.0)["render"]
            uni = render_rgb(cam, gs, pipe, bg, s=args.s)["render"]
            ours = render_rgb(cam, gs, pipe, bg, s=s_vector(h, sb, args.k)[:, None])["render"]
        mask = angle_map(R, d, W, fov) < args.fovea
        uv = project_dir(R, d, W, W, fov, fov)
        pu, po = fovea_psnr(uni, ref, mask), fovea_psnr(ours, ref, mask)
        if pu == pu and po == po:
            gains.append(po - pu)
        a = panel(uni, f"CLoD-GS (uniform, s = {args.s:g})", f"{target:,} Gaussians in view | fovea PSNR {pu:.1f} dB", uv, ring, args.zoom)
        b = panel(ours, f"GazeCLoD-GS (crowd head + gaze heat, k = {args.k:g})", f"{n:,} Gaussians in view | fovea PSNR {po:.1f} dB", uv, ring, args.zoom)
        frame = Image.new("RGB", size, (0, 0, 0))
        frame.paste(a, (0, 0))
        frame.paste(b, (W + 6, 0))
        dr = ImageDraw.Draw(frame)
        dr.text((10, W + 10), f"{args.scene}, NTHU {args.user}, t = {tr['t_ms'][i] / 1000:5.1f} s.  Red circle: where this viewer "
                              f"actually looked (5 deg), inset zoomed 3x. Same number of Gaussians in view.", fill=(220, 220, 220))
        vw.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"Saved {out}  ({len(rows)} frames at {args.fps:g} fps). "
          f"Median fovea gain over the video: {np.median(gains):+.2f} dB")


if __name__ == "__main__":
    main()
