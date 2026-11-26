"""
Session 4 picture: color the splat by crowd heat and report how concentrated it is.

Run from the repo root with the clodgs env active (one or more heat files):
    python gazeclod/show_heat.py -m output/truck_clod --heat output/truck_clod/gaze_heat/heat_rutgers_truck_density_sig2.5.npz
Writes <first heat file>_preview.png: normal render on the left, one heat column per file.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod.common import (base_colors, load_model, lod_pipe, photo_cameras,  # noqa: E402
                             render_colors, render_rgb, to_uint8)


def turbo(x):
    """Polynomial fit of Google's Turbo colormap. x in [0, 1], returns (..., 3)."""
    x = x.clamp(0, 1)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return torch.stack([r, g, b], -1).clamp(0, 1)


def crowd_heat(npz_path, users=None):
    """Crowd heat of one file, or of several joined with '+' (summed, e.g. gaze + head mass)."""
    parts = npz_path.split("+")
    H, names = None, []
    for p in parts:
        z = np.load(p)
        sel = np.ones(len(z["users"]), bool) if not users else np.isin(z["users"], users)
        H = z["heat"][sel].sum(0) if H is None else H + z["heat"][sel].sum(0)
        names.append(str(z["mode"]) if "mode" in z.files else "mass")
        user_list = list(z["users"][sel])
    mode = " + ".join(names) if len(names) > 1 else names[0]
    return H, user_list, mode


def concentration(H):
    """Fraction of all Gaussians needed to hold 50% and 90% of the total heat."""
    s = np.sort(H)[::-1]
    c = np.cumsum(s) / s.sum()
    n = len(H)
    return (np.searchsorted(c, 0.5) + 1) / n, (np.searchsorted(c, 0.9) + 1) / n


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", required=True, nargs="+", help=".npz file(s) written by heatmap.py")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--ids", type=int, nargs="*", default=[0, 10, 20],
                    help="cameras.json ids to render (0-31 are the test views)")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--gamma", type=float, default=0.5, help="linear scale only: <1 brightens weak heat")
    ap.add_argument("--scale", default="auto", choices=["auto", "log", "linear"],
                    help="log: colors span the 50th to 99.9th percentile of positive heat on a log scale. "
                         "auto: linear for density (stamp) heat, log for mass (blame) heat")
    args = ap.parse_args()

    gs, _ = load_model(args.model, args.iteration)
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    gray = base_colors(gs).mean(1, keepdim=True).expand(-1, 3) * 0.5

    columns = []  # (label, per-Gaussian colors)
    for path in args.heat:
        H, users, mode = crowd_heat(path)
        assert len(H) == gs.get_xyz.shape[0], f"{path} was made for a different model"
        f50, f90 = concentration(H)
        print(f"{os.path.basename(path)}: {len(users)} users, {mode} mode. "
              f"{(H > 0).mean():.1%} of Gaussians got any heat. "
              f"Half of all heat sits on {f50:.2%} of Gaussians, 90% on {f90:.2%}.")
        Ht = torch.tensor(H, device="cuda")
        pos = Ht[Ht > 0]
        sub = pos[::max(1, pos.numel() // 1_000_000)]
        scale = args.scale if args.scale != "auto" else ("linear" if mode == "density" else "log")
        if scale == "log":
            lo, hi = torch.log(torch.quantile(sub, 0.5)), torch.log(torch.quantile(sub, 0.999))
            hv = ((torch.log(Ht.clamp_min(1e-20)) - lo) / (hi - lo)).clamp(0, 1)
            hv[Ht <= 0] = 0
        else:
            hv = (Ht / torch.quantile(sub, 0.99)).clamp(0, 1) ** args.gamma
        colors = (1 - hv[:, None]) * gray + hv[:, None] * turbo(hv)
        nice = {"mass": "gaze, blame scoring", "mass_head": "head direction, blame scoring",
                "mass + mass_head": "what GazeCLoD-GS protects (head + gaze, blame scoring)",
                "density": "where eyes landed (stamp scoring, size-free)"}.get(mode, mode)
        columns.append((f"{nice}, {len(users)} people, {scale} scale", colors))

    rows = []
    for cam in photo_cameras(args.model, args.width, ids=args.ids):
        with torch.no_grad():
            tiles = [label(Image.fromarray(to_uint8(render_rgb(cam, gs, pipe, bg)["render"])),
                           f"render  ({cam.image_name})")]
            for text, colors in columns:
                img = to_uint8(render_colors(cam, gs, pipe, bg, colors)["render"])
                tiles.append(label(Image.fromarray(img), text))
        row = Image.new("RGB", (tiles[0].width * len(tiles), tiles[0].height))
        for k, t in enumerate(tiles):
            row.paste(t, (k * t.width, 0))
        rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    first = args.heat[0].split("+")[0]
    out = os.path.splitext(first)[0] + ("_combo" if "+" in args.heat[0] else "") + "_preview.png"
    sheet.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
