"""
Quality-vs-budget figure and gain tables from the CSVs written by eval_gaze_lod.py.

Only needs numpy and matplotlib (no GPU), so it also runs on a laptop without CUDA.

Example (the figure in the README and report):
    python gazeclod/plot_results.py ^
        --scene truck=output/truck_clod/gaze_lod/results_e1b.csv,output/truck_clod/gaze_lod/results_e1c.csv,output/truck_clod/gaze_lod/results_e1d.csv ^
        --scene playroom=output/playroom_clod/gaze_lod/results_e1.csv
(^ continues a line in Windows cmd, use ` in PowerShell, or put it all on one line.)

For each scene, every CSV is read and merged (the same frame and policy found twice keeps the
first copy, so uniform can appear in every file). Series that are missing from a scene are
skipped. Outputs, next to --out:
    <out>.png and <out>.pdf   2 rows (where they look, whole frame) x one column per scene
    <out>_gains.md            median gain vs uniform per scene and budget, Markdown table
    <out>_gains.tex           the same table for LaTeX (booktabs)
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

# policy name -> (legend label, color, marker). Order = legend and drawing order.
SERIES = [
    ("combo_k{k}", "GazeCLoD-GS (head + gaze)", "#257de0", "o"),
    ("uniform", "CLoD-GS (uniform)", "#f47230", "s"),
    ("head_k{k}", "Head direction only", "#00b894", "^"),
    ("gaze_k{k}", "Gaze only", "#f2a600", "D"),
    ("shuffled_k{k}", "Shuffled control", "#ee7fac", "v"),
]
REGIONS = [("mse_fov", "PSNR where the viewer looks (dB)"), ("mse_full", "PSNR over the whole frame (dB)")]


def psnr(mse):
    """Same as eval_gaze_lod.psnr: pixel values in [0, 1], error floored at 1e-10 (100 dB)."""
    return 10 * math.log10(1.0 / max(mse, 1e-10)) if mse == mse else float("nan")


def load(paths):
    """{policy: {(fold, user, row, s): csv row}} merged over several CSVs, first copy wins."""
    data = defaultdict(dict)
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["fold"], r["user"], r["row"], float(r["s"]))
                data[r["policy"]].setdefault(key, r)
    return data


def budgets(data):
    return sorted({key[3] for key in data["uniform"]})


def median_curve(data, policy, region, s_list):
    """Median PSNR over frames, one value per budget."""
    out = []
    for s in s_list:
        vals = [psnr(float(r[region])) for key, r in data[policy].items() if key[3] == s]
        out.append(np.nanmedian(vals) if vals else np.nan)
    return np.array(out)


def in_view_thousands(data, s_list):
    """x axis: median number of Gaussians CLoD-GS keeps in view at each budget (every policy is matched to it)."""
    return np.array([np.median([float(r["drawn_view"]) for key, r in data["uniform"].items() if key[3] == s]) / 1e3
                     for s in s_list])


def gains(data, policy, s):
    """Paired over frames vs uniform: median PSNR gain where they look and over the whole frame, and win rate."""
    d_fov, d_full, wins = [], [], []
    for key, r in data[policy].items():
        u = data["uniform"].get(key)
        if key[3] != s or u is None:
            continue
        mf, uf = float(r["mse_fov"]), float(u["mse_fov"])
        if mf == mf and uf == uf:  # skip frames whose gaze region was too small (NaN)
            d_fov.append(psnr(mf) - psnr(uf))
            wins.append(mf < uf)
        d_full.append(psnr(float(r["mse_full"])) - psnr(float(u["mse_full"])))
    return np.median(d_fov), np.median(d_full), 100 * np.mean(wins), len(d_full)


def plot(scenes, k, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "xtick.color": "#555555", "ytick.color": "#555555"})
    fig, axes = plt.subplots(2, len(scenes), figsize=(5.6 * len(scenes), 8.2), squeeze=False)
    handles = {}
    for c, (name, data) in enumerate(scenes):
        s_list = budgets(data)
        x = in_view_thousands(data, s_list)
        for r, (region, ylabel) in enumerate(REGIONS):
            ax = axes[r][c]
            ax.grid(True, color="#e6e6e6", linewidth=1)
            ax.set_axisbelow(True)
            for pol, label, color, marker in SERIES:
                pol = pol.format(k=f"{k:g}")
                if pol not in data:
                    continue
                y = median_curve(data, pol, region, s_list)
                (h,) = ax.plot(x, y, color=color, linewidth=2.5, marker=marker, markersize=8,
                               markeredgecolor="white", markeredgewidth=1.5, label=label)
                handles.setdefault(label, h)
            if r == 0:
                ax.set_title(name, loc="left", fontweight="bold")
            if r == 1:
                ax.set_xlabel("Gaussians drawn in view (thousands)")
            if c == 0:
                ax.set_ylabel(ylabel)
    n_frames = max(len({key[:3] for key in data["uniform"]}) for _, data in scenes)
    fig.suptitle(f"Same budget, better where people look: median over up to {n_frames} test frames per scene (k = {k:g})",
                 x=0.02, ha="left", fontsize=12)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"wrote {out}.png and {out}.pdf")


def tables(scenes, k, out):
    pol = f"combo_k{k:g}"
    md = ["| Scene | s | Gaussians in view | Where they look | Whole frame | Wins where they look | Frames |",
          "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Scene & $s$ & In view & $\Delta$PSNR look (dB) & $\Delta$PSNR full (dB) & Win (\%) & Frames \\", r"\midrule"]
    for name, data in scenes:
        if pol not in data:
            print(f"{name}: no {pol} rows, skipping its table")
            continue
        s_list = budgets(data)
        for s, x in zip(s_list, in_view_thousands(data, s_list)):
            f, w, win, n = gains(data, pol, s)
            md.append(f"| {name} | {s:g} | {x:.0f}k | {f:+.1f} dB | {w:+.1f} dB | {win:.0f}% | {n} |")
            tex.append(f"{name} & {s:g} & {x:.0f}k & {f:+.1f} & {w:+.1f} & {win:.0f} & {n} \\\\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    open(f"{out}_gains.md", "w").write("\n".join(md) + "\n")
    open(f"{out}_gains.tex", "w").write("\n".join(tex) + "\n")
    print(f"\nMedian gain of {pol} over CLoD-GS (uniform) at the same number of Gaussians in view:")
    print("\n".join(md))
    print(f"wrote {out}_gains.md and {out}_gains.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", action="append", required=True,
                    help="name=results.csv[,more.csv]; repeat for each scene (one figure column each)")
    ap.add_argument("--k", type=float, default=4.0, help="strength whose policies are plotted (combo_k4, head_k4, ...)")
    ap.add_argument("--out", default="output/figures/quality_vs_budget", help="output path without extension")
    args = ap.parse_args()

    scenes = []
    for spec in args.scene:
        name, files = spec.split("=", 1)
        data = load(files.split(","))
        if "uniform" not in data:
            raise SystemExit(f"{name}: no 'uniform' rows in {files}")
        missing = [p.format(k=f"{args.k:g}") for p, *_ in SERIES if p.format(k=f"{args.k:g}") not in data]
        if missing:
            print(f"{name}: not in these CSVs, skipped: {', '.join(missing)}")
        scenes.append((name.capitalize(), data))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(scenes, args.k, args.out)
    tables(scenes, args.k, args.out)


if __name__ == "__main__":
    main()
