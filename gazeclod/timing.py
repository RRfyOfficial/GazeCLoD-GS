"""
Render-time test at larger, VR-like image sizes.

The main experiment (eval_gaze_lod.py) renders at 512 x 512, where a big part of every frame's cost
does not depend on how many Gaussians are drawn. A VR headset renders much bigger images (a Quest Pro
eye is about 1800 x 1920), so this script checks whether the Gaussians GazeCLoD-GS saves turn into a
bigger time saving there.

For one moment per NTHU viewer (the middle of their session, heat from the other team as in the main
experiment), and for every image size and budget s, it times:
  uniform    plain CLoD-GS at dial s
  same       GazeCLoD-GS with the same number of Gaussians in view (should take the same time)
  equal      GazeCLoD-GS with fewer Gaussians in view: the share it needs to match plain CLoD-GS's
             quality where people look, measured at 512 x 512 in the main experiment (--savings)
plus "full" (s = 1) as a reference. Each render is timed with CUDA events: --warmup runs, then the
median of --repeats runs, with the methods interleaved so clock changes hit all of them equally.
It also times the only extra per-frame work GazeCLoD-GS needs, computing every s_i (the budget
search is only there to make the comparison fair).

Close other GPU programs and keep a laptop plugged in, or the numbers will be noisy.
Run from the repo root, e.g.:
    python gazeclod/timing.py -m output/truck_clod --scene truck --heat <gaze heat>+<head heat>
Writes <model>/gaze_lod/timing_<tag>.csv and prints (and saves) a summary table.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import make_minicam  # noqa: E402
from gazeclod.common import load_model, lod_pipe, render_rgb  # noqa: E402
from gazeclod.eval_gaze_lod import HeatBank  # noqa: E402
from gazeclod.heatmap import fixation_rows  # noqa: E402
from gazeclod.lod_policy import LodState, match_budget, s_vector, user_number  # noqa: E402

# Share of Gaussians in view GazeCLoD-GS (head + gaze, k = 4) can drop and still match plain CLoD-GS's
# median quality where people look, from the 512 x 512 experiment (budgets where it could be measured).
SAVINGS = {"truck": "4:0.13,8:0.13,12:0.23,16:0.35", "playroom": "8:0.23,12:0.30,16:0.32"}


def cuda_ms(fn, keep=False):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    out = fn()
    b.record()
    torch.cuda.synchronize()
    return (a.elapsed_time(b), out) if keep else a.elapsed_time(b)


def screen_work(radii, size):
    """
    How much screen the drawn Gaussians cover, from the rasterizer's per-Gaussian radius r (pixels):
      tiles = sum of (2r/16 + 1)^2, about how many 16x16 tiles each Gaussian touches. The rasterizer
              blends every Gaussian into every pixel of every tile it touches, so this tracks its work.
      cover = sum of pi r^2 / pixels, about how many Gaussians lie over an average pixel.
    """
    r = radii[radii > 0].float()
    return float(((2 * r / 16 + 1) ** 2).sum()), float((math.pi * r ** 2).sum() / (size * size))


def test_moments(bank, eyenav, site, scene):
    """One moment per viewer: the middle steady glance of their session, with heat from the other team."""
    traces = {G.user_id(p): p for p in G.list_traces(eyenav, site, scene)}
    out = []
    for uid in bank.users:
        tr = G.to_scene(G.load_trace(traces[uid]))
        rows, _ = fixation_rows(tr, 0.5, 100.0)
        i = int(rows[len(rows) // 2])
        heat_team = [u for u in bank.users if user_number(u) % 2 != user_number(uid) % 2]
        out.append((uid, i, tr["R_c2w"][i], tr["C"][i], heat_team))
    return out


def summarize(csv_path, out_md=None):
    rows = list(csv.DictReader(open(csv_path, newline="")))
    by = defaultdict(list)
    for r in rows:
        by[(int(r["size"]), float(r["s"]), r["policy"])].append(r)
    lines = []
    for size in sorted({k[0] for k in by}):
        uni = [(float(r["n_view"]), float(r["ms"])) for k, g in by.items() if k[0] == size and k[2] in ("uniform", "full")
               for r in g if r["ms"] != "nan"]
        if len(uni) < 2:
            continue
        n, ms = np.array(uni).T
        b, a = np.polyfit(n, ms, 1)
        lines.append(f"\n### {size} x {size}\n")
        lines.append(f"Plain CLoD-GS render time is about {a:.1f} ms + {b * 1e5:.2f} ms per 100k Gaussians in view "
                     f"(straight-line fit over every uniform render).\n")
        lines.append("| s | In view, uniform | Uniform ms | Same budget ms | Equal quality: in view | Equal quality ms | Time saved |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in sorted({k[1] for k in by if k[0] == size}):
            def med(pol, col="ms"):
                g = [float(r[col]) for r in by.get((size, s, pol), []) if r[col] != "nan"]
                return np.median(g) if g else float("nan")
            u, same, eq = med("uniform"), med("same"), med("equal")
            if s == 1:
                lines.append(f"| 1 (full) | {med('full', 'n_view') / 1e3:.0f}k | {med('full'):.2f} | | | | |")
                continue
            saved = f"{100 * (1 - eq / u):.0f}%" if eq == eq else ""
            eq_n = f"{med('equal', 'n_view') / 1e3:.0f}k" if eq == eq else ""
            eq_ms = f"{eq:.2f}" if eq == eq else ""
            lines.append(f"| {s:g} | {med('uniform', 'n_view') / 1e3:.0f}k | {u:.2f} | {same:.2f} | {eq_n} | {eq_ms} | {saved} |")
    if rows and rows[0].get("tiles") not in (None, ""):
        pair = defaultdict(dict)
        for r in rows:
            pair[(int(r["size"]), r["user"], float(r["s"]))][r["policy"]] = r
        lines.append("\n### Same number of Gaussians in view: how much more work GazeCLoD-GS's Gaussians make\n")
        lines.append("| Size | s | Extra time | Extra tile work | Avg. Gaussians over a pixel, plain / ours |")
        lines.append("|---|---|---|---|---|")
        for size in sorted({k[0] for k in pair}):
            for s in sorted({k[2] for k in pair if k[0] == size and k[2] != 1}):
                g = [v for k, v in pair.items() if k[0] == size and k[2] == s and "same" in v and "uniform" in v]
                if not g:
                    continue
                t = np.median([float(v["same"]["ms"]) / float(v["uniform"]["ms"]) for v in g])
                w = np.median([float(v["same"]["tiles"]) / float(v["uniform"]["tiles"]) for v in g])
                cu = np.median([float(v["uniform"]["cover"]) for v in g])
                co = np.median([float(v["same"]["cover"]) for v in g])
                lines.append(f"| {size} | {s:g} | {100 * (t - 1):+.1f}% | {100 * (w - 1):+.1f}% | {cu:.1f} / {co:.1f} |")
        lines.append("\nMedians over test moments of the per-moment ratio (GazeCLoD-GS / plain CLoD-GS). Tile work = sum over "
                     "drawn Gaussians of the 16x16 tiles each one touches.")
    sv = [float(r["sv_ms"]) for r in rows if r.get("sv_ms") not in (None, "", "nan")]
    if sv:
        lines.append(f"\nComputing every s_i (GazeCLoD-GS's only extra per-frame work): median {np.median(sv):.3f} ms.")
    lines.append("\nMedians over test moments. 'Time saved' compares the equal-quality GazeCLoD-GS render with plain "
                 "CLoD-GS at the same s, assuming the 512 x 512 equal-quality share carries over to this size.")
    text = "\n".join(lines)
    print(text)
    if out_md:
        open(out_md, "w").write(text + "\n")
        print(f"\nwrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Render-time test at larger image sizes (see the top of this file).")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", help="heat .npz with per-user rows, 'a.npz+b.npz' sums files (gaze + head)")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 1536, 2048])
    ap.add_argument("--s", type=float, nargs="+", default=[2, 4, 8, 12, 16])
    ap.add_argument("--savings", default=None, help="s:share pairs, e.g. '8:0.23,12:0.30' (default: this scene's)")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-users", type=int, default=0, help="only the first N viewers (quick test)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--summary", help="only print the summary of this timing CSV and exit")
    args = ap.parse_args()

    if args.summary:
        summarize(args.summary)
        return
    if not args.heat:
        sys.exit("--heat is required")
    savings = args.savings or SAVINGS.get(args.scene, "")
    savings = {float(a): float(b) for a, b in (p.split(":") for p in savings.split(",") if p)}

    bank = HeatBank(args.heat)
    gs, _ = load_model(args.model, args.iteration)
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    moments = test_moments(bank, args.eyenav, args.site, args.scene)[: args.max_users or None]
    h_cache = {}
    fov = math.radians(args.fov)

    out_dir = os.path.join(args.model, "gaze_lod")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"timing_{args.tag}.csv")
    fields = ["size", "user", "row", "s", "policy", "s_base", "n_view", "ms", "sv_ms", "tiles", "cover"]
    fh = open(out_csv, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    print(f"{len(moments)} test moments x {len(args.sizes)} sizes x {len(args.s)} budgets. Savings used: {savings}")

    for size in args.sizes:
        torch.cuda.empty_cache()
        print(f"\n== {size} x {size}")
        for n_m, (uid, i, R, C, heat_team) in enumerate(moments):
            key = tuple(heat_team)
            if key not in h_cache:
                h_cache[key] = bank.h(heat_team, args.gamma)
            h = h_cache[key]
            cam = make_minicam(R, C, size, size, fov, fov)
            st = LodState(gs, cam)
            jobs = [("full", 1.0, 1.0, st.count(1.0), None)]
            for s in args.s:
                target = st.count(s)
                sb, n = match_budget(st, h, args.k, target, s)
                jobs.append(("uniform", s, s, target, None))
                jobs.append(("same", s, sb, n, s_vector(h, sb, args.k)[:, None]))
                if s in savings:
                    sb2, n2 = match_budget(st, h, args.k, int(target * (1 - savings[s])), s)
                    jobs.append(("equal", s, sb2, n2, s_vector(h, sb2, args.k)[:, None]))
            times, work = defaultdict(list), {}
            try:
                with torch.no_grad():
                    for rep in range(args.warmup + args.repeats):
                        for j, (pol, s, sb, n, sv) in enumerate(jobs):
                            ms, pkg = cuda_ms(lambda: render_rgb(cam, gs, pipe, bg, s=sv if sv is not None else s), keep=True)
                            if rep >= args.warmup:
                                times[j].append(ms)
                            if rep == args.warmup:  # same Gaussians every repeat, so measure coverage once
                                work[j] = screen_work(pkg["radii"], size)
                            del pkg
                    sv_ms = np.median([cuda_ms(lambda: s_vector(h, jobs[-1][2], args.k)) for _ in range(10)])
            except torch.cuda.OutOfMemoryError:
                print(f"  out of GPU memory at {size} x {size}, skipping this size")
                torch.cuda.empty_cache()
                break
            for j, (pol, s, sb, n, sv) in enumerate(jobs):
                writer.writerow({"size": size, "user": uid, "row": i, "s": s, "policy": pol,
                                 "s_base": round(float(sb), 4), "n_view": n,
                                 "ms": round(float(np.median(times[j])), 3), "sv_ms": round(float(sv_ms), 4),
                                 "tiles": round(work[j][0]), "cover": round(work[j][1], 3)})
            fh.flush()
            ju = max(j for j, job in enumerate(jobs) if job[0] == "uniform")
            print(f"  {n_m + 1}/{len(moments)} {uid}: full detail {np.median(times[0]):.1f} ms, "
                  f"uniform s={jobs[ju][1]:g} {np.median(times[ju]):.1f} ms", end="\r")
        print()
    fh.close()
    print(f"\nResults: {out_csv}")
    summarize(out_csv, out_csv.replace(".csv", "_summary.md"))


if __name__ == "__main__":
    main()
