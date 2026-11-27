"""
Demo video: replay one real NTHU viewer's 60 s walk through the splat, plain CLoD-GS and
GazeCLoD-GS side by side at the SAME number of Gaussians in view. The red circle is where this
viewer was actually looking (their own eye tracking) and the inset zooms in on it. The heat comes
from the other team, so this viewer never contributed to it.

Each side shows its quality where the viewer looks (PSNR against the full-detail render, in dB,
higher = closer to full detail), and a running chart under the two views plots both numbers over
the whole session, so the gap is visible even when the images look alike at a glance.

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


ORANGE, BLUE, GREEN, RED = (244, 114, 48), (90, 160, 240), (70, 200, 120), (255, 80, 80)
HEADER, CHART, FOOTER = 64, 150, 50


def font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_MED, F_SMALL = font(22), font(17), font(14)


def panel(img_chw, title, n, q, accent, uv, ring, zoom, gain=None, scale=3):
    """One side: the render with the gaze circle and zoom inset, and a header with its quality where the viewer looks."""
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
        iy = HEADER + 6 if v > W / 2 else W - inset.height - 8
        im.paste(inset, (ix, iy))
        d.rectangle([ix - 2, iy - 2, ix + inset.width + 1, iy + inset.height + 1], outline=(255, 40, 40), width=2)
    d.rectangle([0, 0, W, HEADER], fill=(0, 0, 0))
    d.rectangle([0, 0, 6, HEADER], fill=accent)
    d.text((16, 6), title, fill=accent, font=F_BIG)
    d.text((16, 38), f"{n:,} Gaussians in view", fill=(200, 200, 200), font=F_SMALL)
    qtxt = f"{q:.1f} dB" if q == q else "n/a"
    lab = "where they look:"
    xq = W - 12 - d.textlength(qtxt, font=F_BIG)
    d.text((xq, 6), qtxt, fill=(255, 255, 255), font=F_BIG)
    d.text((xq - 8 - d.textlength(lab, font=F_SMALL), 12), lab, fill=(200, 200, 200), font=F_SMALL)
    if gain is not None and gain == gain:
        gtxt = f"{gain:+.1f} dB vs plain"
        d.text((W - 12 - d.textlength(gtxt, font=F_MED), 38), gtxt, fill=GREEN if gain >= 0 else RED, font=F_MED)
    return im


def chart(width, t_all, t_now, hist_u, hist_o, lo=15.0, hi=55.0):
    """Running chart of quality where the viewer looks, both methods, over the whole session."""
    im = Image.new("RGB", (width, CHART), (16, 16, 16))
    d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = 64, width - 20, 28, CHART - 22
    t0, t1 = t_all[0], t_all[-1]

    def xy(t, q):
        q = min(max(q, lo), hi)
        return (x0 + (x1 - x0) * (t - t0) / max(t1 - t0, 1e-6), y1 - (y1 - y0) * (q - lo) / (hi - lo))

    for g in range(int(lo // 10 + 1) * 10, int(hi) + 1, 10):
        _, y = xy(t0, g)
        d.line([(x0, y), (x1, y)], fill=(55, 55, 55), width=1)
        d.text((x0 - 8 - d.textlength(f"{g}", font=F_SMALL), y - 8), f"{g}", fill=(150, 150, 150), font=F_SMALL)
    for sec in range(0, int(t1 - t0) + 1, 10):
        x, _ = xy(t0 + sec, lo)
        d.text((x - 8, y1 + 3), f"{sec}s", fill=(150, 150, 150), font=F_SMALL)
    for hist, color in ((hist_u, ORANGE), (hist_o, BLUE)):
        seg = []
        for t, q in hist:
            if q == q:
                seg.append(xy(t, q))
            elif len(seg) > 1:
                d.line(seg, fill=color, width=2)
                seg = []
            else:
                seg = []
        if len(seg) > 1:
            d.line(seg, fill=color, width=2)
    xn, _ = xy(t_now, lo)
    d.line([(xn, y0), (xn, y1)], fill=(200, 200, 200), width=1)
    d.text((x0, 5), "Quality where the viewer looks (PSNR vs full detail, dB, higher is better):", fill=(220, 220, 220), font=F_SMALL)
    xl = x0 + d.textlength("Quality where the viewer looks (PSNR vs full detail, dB, higher is better):", font=F_SMALL) + 14
    for label, color in (("plain CLoD-GS", ORANGE), ("GazeCLoD-GS", BLUE)):
        d.line([(xl, 13), (xl + 22, 13)], fill=color, width=3)
        d.text((xl + 28, 5), label, fill=color, font=F_SMALL)
        xl += 40 + d.textlength(label, font=F_SMALL)
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
    size = (2 * W + 6, W + CHART + FOOTER)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    gains, hist_u, hist_o = [], [], []
    t_all = tr["t_ms"][rows] / 1000.0
    t_all = t_all - t_all[0]
    for j, i in enumerate(tqdm(rows, desc="frames")):
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
        hist_u.append((t_all[j], pu))
        hist_o.append((t_all[j], po))
        if pu == pu and po == po:
            gains.append(po - pu)
        a = panel(uni, f"Plain CLoD-GS (s = {args.s:g})", target, pu, ORANGE, uv, ring, args.zoom)
        b = panel(ours, f"GazeCLoD-GS (ours, k = {args.k:g})", n, po, BLUE, uv, ring, args.zoom, gain=po - pu)
        frame = Image.new("RGB", size, (0, 0, 0))
        frame.paste(a, (0, 0))
        frame.paste(b, (W + 6, 0))
        frame.paste(chart(size[0], t_all, t_all[j], hist_u, hist_o), (0, W))
        dr = ImageDraw.Draw(frame)
        dr.text((10, W + CHART + 5), f"{args.scene}, NTHU {args.user}, t = {t_all[j]:4.1f} s.   Red circle: where this viewer "
                f"actually looked (5\u00b0).   Corner box: 3x zoom.   Same number of Gaussians on both sides.",
                fill=(220, 220, 220), font=F_SMALL)
        if gains:
            dr.text((10, W + CHART + 26), f"So far: GazeCLoD-GS is sharper where they look in "
                    f"{100 * np.mean(np.array(gains) > 0):.0f}% of frames, median {np.median(gains):+.1f} dB.",
                    fill=GREEN, font=F_SMALL)
        vw.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"Saved {out}  ({len(rows)} frames at {args.fps:g} fps). "
          f"Median fovea gain over the video: {np.median(gains):+.2f} dB")


if __name__ == "__main__":
    main()
