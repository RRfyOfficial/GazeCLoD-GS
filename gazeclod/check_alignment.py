"""
Session 3 sanity check: does EyeNavGS gaze line up with our trained splat?

Renders the splat from one participant's recorded left-eye poses and draws a
red ring where they were looking (ring radius = 2.5 deg, about the fovea).
If the views look upright and natural and the rings sit on real things
(the truck, wheels, signs, plants), the alignment works.

Run from the repo root, in the terminal where the gazeclod env is active:
    python gazeclod/check_alignment.py -m output/truck_clod --scene truck --site nthu --user user10
    python gazeclod/check_alignment.py -m output/truck_clod --scene truck --site rutgers --user user102

Writes  <model>/gaze_check/align_<site>_<user>.png
"""
import argparse
import os
import sys
from argparse import Namespace  # noqa: F401  (needed to eval cfg_args)

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import focal, make_minicam, project_dir  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True
except Exception:
    SEPARATE_SH = False


def load_model(model_dir, iteration):
    with open(os.path.join(model_dir, "cfg_args")) as f:
        cfg = eval(f.read())  # 3DGS saves its args as a Namespace(...) string
    gs = GaussianModel(cfg.sh_degree, enable_opacity_decay_train=True)
    gs.load_ply(os.path.join(model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    return gs, cfg


def lod_pipe():
    # Same switches render.py uses: decayed opacity + distance culling
    return Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False,
                     antialiasing=False, enable_decay_opacity_render=True,
                     enable_distance_culling=True, opacity_threshold=0.005,
                     enable_view_distance=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="e.g. output/truck_clod")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu", choices=list(G.SITES))
    ap.add_argument("--user", default=None, help="e.g. user10 (default: first user)")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--size", type=int, default=640, help="square image size in pixels")
    ap.add_argument("--fov", type=float, default=90.0, help="symmetric field of view in degrees")
    ap.add_argument("--s", type=float, default=1.0, help="CLoD virtual distance scale")
    args = ap.parse_args()

    paths = G.list_traces(args.eyenav, args.site, args.scene)
    if not paths:
        sys.exit(f"No traces found for {args.site}/{args.scene} under {args.eyenav}")
    if args.user:
        paths = [p for p in paths if G.user_id(p) == args.user]
        if not paths:
            sys.exit(f"{args.user} not found for {args.site}/{args.scene}")
    tr = G.load_trace(paths[0])
    if len(tr["view"]) == 0:
        sys.exit(f"{G.user_id(paths[0])} has no valid gaze rows, pick another --user")
    tr = G.to_scene(tr)
    user = G.user_id(paths[0])

    ecc = G.eccentricity_deg(tr)
    print(f"{args.site} {user}: {len(tr['view']) // 2} frames, "
          f"gaze-to-head angle median {np.median(ecc):.1f} deg, p95 {np.percentile(ecc, 95):.1f} deg")

    gs, _ = load_model(args.model, args.iteration)
    print(f"Loaded {gs.get_xyz.shape[0]:,} Gaussians")
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    W = H = args.size
    fov = np.radians(args.fov)
    ring = focal(fov, W) * np.tan(np.radians(2.5))

    left = np.where(tr["view"] == 0)[0]
    picks = left[np.linspace(0, len(left) - 1, args.frames).astype(int)]
    tiles = []
    for i in picks:
        cam = make_minicam(tr["R_c2w"][i], tr["C"][i], W, H, fov, fov)
        with torch.no_grad():
            pkg = render(cam, gs, pipe, bg, virtual_scale=args.s, separate_sh=SEPARATE_SH)
        img = (pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        uv = project_dir(tr["R_c2w"][i], tr["gaze_d"][i], W, H, fov, fov)
        if uv is not None:
            u, v = uv
            d.ellipse([u - ring, v - ring, u + ring, v + ring], outline=(255, 40, 40), width=3)
            d.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 40, 40))
        used = pkg.get("points_used", 0)
        d.rectangle([0, 0, W, 22], fill=(0, 0, 0))
        d.text((6, 5), f"{args.site} {user}  t={tr['t_ms'][i] / 1000:.1f}s  "
                       f"drawn {used:,} Gaussians", fill=(255, 255, 255))
        tiles.append(im)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H), "black")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * W, (k // cols) * H))
    out_dir = os.path.join(args.model, "gaze_check")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"align_{args.site}_{user}.png")
    sheet.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
