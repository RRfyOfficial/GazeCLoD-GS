"""
Session 5 check: at the same number of Gaussians, does the gaze knob look
better where people actually look?

Heat comes from one NTHU team, test frames from the other team's real head
poses, and the foveal score uses that viewer's real gaze. For each frame we
render four images:
  reference  s = 1 (full detail, the target)
  uniform    plain CLoD-GS at scale --s
  gaze       per-Gaussian knob from the crowd heat, budget-matched to uniform
  shuffled   the same heat values on random Gaussians, budget-matched (control)
PSNR against the reference is reported over the full frame, the fovea
(within --fovea degrees of the viewer's gaze) and the periphery.

Run from the repo root with the clodgs env active:
    python gazeclod/lod_demo.py -m output/truck_clod --heat output/truck_clod/gaze_heat/heat_nthu_truck_density_sig2.5.npz
Writes <model>/gaze_lod/demo_<test team>_s<s>_k<k>.png
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
from gazeclod.lod_policy import (LodState, heat_to_h, match_budget, s_vector,  # noqa: E402
                                 shuffled, split_teams, team_heat)


def psnr(a, b, mask=None):
    """a, b: (3, H, W) tensors in [0, 1]. mask: (H, W) bool numpy or None."""
    err = ((a - b) ** 2).mean(0)
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        err = err[torch.as_tensor(mask, device=err.device)]
    mse = max(float(err.mean()), 1e-10)
    return 10 * math.log10(1.0 / mse)


def tile(img, text, ring):
    im = Image.fromarray(to_uint8(img))
    d = ImageDraw.Draw(im)
    if ring is not None:
        (u, v), r = ring
        d.ellipse([u - r, v - r, u + r, v + r], outline=(255, 40, 40), width=2)
    d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", required=True, help="NTHU density heat .npz from heatmap.py")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--test-team", default="even", choices=["even", "odd"],
                    help="team whose views we test on; the heat comes from the other team")
    ap.add_argument("--s", type=float, default=8.0, help="uniform CLoD scale that sets the budget")
    ap.add_argument("--k", type=float, default=2.0, help="gaze strength, 0 = plain CLoD-GS")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--users-max", type=int, default=3)
    ap.add_argument("--frames-per-user", type=int, default=2)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fovea", type=float, default=5.0, help="foveal radius in degrees")
    args = ap.parse_args()

    odd, even = split_teams(args.heat)
    heat_users, test_users = (odd, even) if args.test_team == "even" else (even, odd)
    H, used = team_heat(args.heat, heat_users)
    print(f"Heat from {len(used)} people ({', '.join(used)}). Testing on {args.test_team} team.")

    gs, _ = load_model(args.model, args.iteration)
    n_total = gs.get_xyz.shape[0]
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    h = heat_to_h(H, args.gamma)
    h_shuf = shuffled(h)
    print(f"h > 0.5 for {(h > 0.5).float().mean().item():.2%} of Gaussians. "
          f"uniform s = {args.s:g}, k = {args.k:g}")

    W = args.size
    fov = math.radians(args.fov)
    ring_px = focal(fov, W) * math.tan(math.radians(args.fovea))
    traces = {G.user_id(p): p for p in G.list_traces(args.eyenav, args.site, args.scene)}

    rows, results = [], []
    for uid in test_users[: args.users_max]:
        tr = G.to_scene(G.load_trace(traces[uid]))
        left = np.where(tr["view"] == 0)[0]
        picks = left[(np.linspace(0.15, 0.85, args.frames_per_user) * (len(left) - 1)).astype(int)]
        for i in picks:
            R, C, d = tr["R_c2w"][i], tr["C"][i], tr["gaze_d"][i]
            cam = make_minicam(R, C, W, W, fov, fov)
            st = LodState(gs, cam)
            target = st.count(args.s)
            sb_g, n_g = match_budget(st, h, args.k, target, args.s)
            sb_r, n_r = match_budget(st, h_shuf, args.k, target, args.s)
            with torch.no_grad():
                ref = render_rgb(cam, gs, pipe, bg, s=1.0)
                uni = render_rgb(cam, gs, pipe, bg, s=args.s)
                gaz = render_rgb(cam, gs, pipe, bg, s=s_vector(h, sb_g, args.k)[:, None])
                shf = render_rgb(cam, gs, pipe, bg, s=s_vector(h_shuf, sb_r, args.k)[:, None])
            for name, pkg, want in [("uniform", uni, st.count_global(args.s)),
                                    ("gaze", gaz, st.count_global(s_vector(h, sb_g, args.k))),
                                    ("shuffled", shf, st.count_global(s_vector(h_shuf, sb_r, args.k)))]:
                if pkg["points_used"] != want:
                    print(f"  warning: {name} kept {pkg['points_used']:,}, predicted {want:,}")

            ang = angle_map(R, d, W, fov)
            fov_mask, per_mask = ang < args.fovea, ang >= args.fovea
            r = {"user": uid, "t": tr["t_ms"][i] / 1000, "budget": target}
            for name, pkg in [("uniform", uni), ("gaze", gaz), ("shuffled", shf)]:
                r[name] = (psnr(pkg["render"], ref["render"]),
                           psnr(pkg["render"], ref["render"], fov_mask),
                           psnr(pkg["render"], ref["render"], per_mask))
            results.append(r)
            print(f"  {uid} t={r['t']:.0f}s  in view {target:,}  "
                  f"fovea PSNR uniform {r['uniform'][1]:.2f} | gaze {r['gaze'][1]:.2f} | "
                  f"shuffled {r['shuffled'][1]:.2f}   (s_base gaze {sb_g:.1f}, shuffled {sb_r:.1f})")

            uv = project_dir(R, d, W, W, fov, fov)
            ring = (uv, ring_px) if uv is not None else None
            tiles = [tile(ref["render"], f"reference s=1  ({ref['points_used']:,})", ring),
                     tile(uni["render"], f"uniform s={args.s:g}  ({target:,} in view)  fovea {r['uniform'][1]:.1f} dB", ring),
                     tile(gaz["render"], f"gaze k={args.k:g}  ({n_g:,})  fovea {r['gaze'][1]:.1f} dB", ring),
                     tile(shf["render"], f"shuffled  ({n_r:,})  fovea {r['shuffled'][1]:.1f} dB", ring)]
            row = Image.new("RGB", (W * 4, W))
            for j, t in enumerate(tiles):
                row.paste(t, (j * W, 0))
            rows.append(row)

    print("\nMean PSNR vs full detail (dB)      full    fovea   periphery")
    for name in ["uniform", "gaze", "shuffled"]:
        m = np.nanmean(np.array([r[name] for r in results]), axis=0)
        print(f"  {name:<10}                      {m[0]:6.2f}  {m[1]:6.2f}  {m[2]:6.2f}")
    mean_budget = np.mean([r["budget"] for r in results])
    print(f"  budget: {mean_budget:,.0f} Gaussians in view on average (model has {n_total:,})")

    out_dir = os.path.join(args.model, "gaze_lod")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"demo_{args.test_team}_s{args.s:g}_k{args.k:g}.png")
    sheet = Image.new("RGB", (W * 4, W * len(rows)))
    for k, row in enumerate(rows):
        sheet.paste(row, (0, k * W))
    sheet.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
