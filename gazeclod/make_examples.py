"""
Session 7 figure: side-by-side example frames at the same Gaussian budget.

For each chosen test frame (fold:user:row, taken from the results CSV, picked
because their gains sit at the median rather than being the best ones):
  full detail (s = 1) | CLoD-GS uniform at s | GazeCLoD-GS at the same budget
Each panel shows where that viewer was actually looking (red circle, --fovea
deg) and a zoomed inset of the fovea. Labels give PSNR in the fovea against
the full-detail render and the number of Gaussians in view.

Run from the repo root with the clodgs env active, e.g.:
    python gazeclod/make_examples.py -m output/truck_clod --scene truck --s 16 --frames even:user12:3538 odd:user19:2894 --heat output/truck_clod/gaze_heat/heat_nthu_truck_mass_sig2.5.npz+output/truck_clod/gaze_heat/heat_nthu_truck_mass_head_sig2.5.npz
Writes <model>/gaze_lod/examples_s<s>_k<k>.png
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import angle_map, focal, make_minicam, project_dir  # noqa: E402
from gazeclod.common import load_model, lod_pipe, render_rgb, to_uint8  # noqa: E402
from gazeclod.eval_gaze_lod import HeatBank  # noqa: E402
from gazeclod.lod_policy import LodState, match_budget, s_vector, user_number  # noqa: E402


def fovea_psnr(img, ref, mask):
    err = ((img - ref) ** 2).mean(0)[torch.as_tensor(mask, device=img.device)]
    return 10 * math.log10(1.0 / max(float(err.mean()), 1e-10))


def panel(img_chw, title, sub, uv, ring, zoom, scale=3):
    im = Image.fromarray(to_uint8(img_chw))
    W = im.width
    d = ImageDraw.Draw(im)
    if uv is not None:
        u, v = uv
        # zoomed inset of the fovea, bottom-right corner
        x0 = int(min(max(round(u) - zoom // 2, 0), W - zoom))
        y0 = int(min(max(round(v) - zoom // 2, 0), W - zoom))
        inset = im.crop((x0, y0, x0 + zoom, y0 + zoom)).resize((zoom * scale, zoom * scale), Image.NEAREST)
        d.ellipse([u - ring, v - ring, u + ring, v + ring], outline=(255, 40, 40), width=3)
        d.rectangle([x0, y0, x0 + zoom, y0 + zoom], outline=(255, 255, 255), width=1)
        ix = 6 if u > W / 2 else W - inset.width - 6   # corner opposite the gaze
        iy = 46 if v > W / 2 else W - inset.height - 6
        im.paste(inset, (ix, iy))
        d.rectangle([ix - 2, iy - 2, ix + inset.width + 1, iy + inset.height + 1], outline=(255, 255, 255), width=2)
    d.rectangle([0, 0, W, 40], fill=(0, 0, 0))
    d.text((8, 5), title, fill=(255, 255, 255))
    d.text((8, 22), sub, fill=(200, 200, 200))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", required=True, help="heat .npz, 'a.npz+b.npz' sums files")
    ap.add_argument("--frames", nargs="+", required=True, help="fold:user:row, fold = team tested")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--s", type=float, default=16.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fovea", type=float, default=5.0)
    ap.add_argument("--zoom", type=int, default=64, help="inset crop size in pixels")
    args = ap.parse_args()

    bank = HeatBank(args.heat)
    gs, _ = load_model(args.model, args.iteration)
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    W, fov = args.size, math.radians(args.fov)
    ring = focal(fov, W) * math.tan(math.radians(args.fovea))
    traces = {G.user_id(p): p for p in G.list_traces(args.eyenav, args.site, args.scene)}
    h_cache = {}

    rows = []
    for spec in args.frames:
        fold, uid, row = spec.split(":")
        row = int(row)
        parity = 0 if fold == "even" else 1  # tested team; heat comes from the other one
        heat_team = [u for u in bank.users if user_number(u) % 2 != parity]
        if fold not in h_cache:
            h_cache[fold] = bank.h(heat_team, args.gamma)
        h = h_cache[fold]
        tr = G.to_scene(G.load_trace(traces[uid]))  # same row indexing as eval_gaze_lod.py
        R, C, d = tr["R_c2w"][row], tr["C"][row], tr["gaze_d"][row]
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
        p_uni, p_ours = fovea_psnr(uni, ref, mask), fovea_psnr(ours, ref, mask)
        n_full = LodState(gs, cam).count(1.0)
        tiles = [panel(ref, "Full detail (s = 1)", f"{n_full:,} Gaussians in view", uv, ring, args.zoom),
                 panel(uni, f"CLoD-GS, uniform s = {args.s:g}", f"{target:,} in view | fovea PSNR {p_uni:.1f} dB", uv, ring, args.zoom),
                 panel(ours, f"GazeCLoD-GS (crowd head + gaze, k = {args.k:g})", f"{n:,} in view | fovea PSNR {p_ours:.1f} dB", uv, ring, args.zoom)]
        print(f"{spec}: fovea PSNR uniform {p_uni:.2f}, ours {p_ours:.2f}, in view {target:,} vs {n:,}")
        strip = Image.new("RGB", (W * 3 + 8, W), "white")
        for j, t in enumerate(tiles):
            strip.paste(t, (j * (W + 4), 0))
        rows.append(strip)

    sheet = Image.new("RGB", (rows[0].width, len(rows) * (W + 4) - 4), "white")
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * (W + 4)))
    out = os.path.join(args.model, "gaze_lod", f"examples_s{args.s:g}_k{args.k:g}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
