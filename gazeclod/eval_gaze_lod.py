"""
Session 6: the experiments. Quality vs Gaussian budget for plain CLoD-GS
(uniform), the gaze knob, and controls, scored where the viewer looks.

Protocol (within-site, 2-fold by people, decided Nov 26):
  NTHU users are split into an odd-numbered and an even-numbered team. The
  heat comes from one team. Test frames come from the other team's real head
  poses, about one fixation per second (saccades skipped), and the foveal
  score uses that viewer's own gaze at that moment. Then the teams swap.

For every test frame and every uniform scale s in --s:
  target = Gaussians uniform CLoD-GS keeps IN VIEW at s (--budget view, the
           default). CLoD-GS's own points_used also counts kept Gaussians
           behind the camera, which the rasterizer discards almost for free.
           A crowd heatmap protects hot Gaussians all around the scene, so a
           global count would charge it for hot Gaussians the viewer cannot
           see (measured: 100k to 170k of them in many frames).
  every other policy is budget-matched to target (never more in view)
  each image is compared with the full-detail render (s = 1)
  logged: in-view count, rasterizer count (radii > 0), global points_used

Policies:
  uniform            plain CLoD-GS
  gaze_k<k>          per-Gaussian knob from the other team's heat (E1)
  shuffled_k<k>      same heat values on random Gaussians (E2 control)
  gaze_k<x>          extra strengths from --k-sweep (E3)
  gaze_n<N>_r<r>     heat from a random N-person subset of the team (E4)
  <name>_k<k>        --extra-heat name=path: same team split, heat from another
                     file of the same site (e.g. mass-mode heat), for ablations
  cross_k<k>         heat from a different file, e.g. Rutgers (optional)
  live_k<k>          heat from THIS viewer's own gaze at THIS moment, same
                     density formula as the crowd heat but one sample (what a
                     headset with an eye tracker could feed the same knob).
                     Off with --no-live
  oracle_k<k>        ceiling: protect exactly the Gaussians that make up the
                     viewer's foveal pixels in this frame, i.e. each
                     Gaussian's foveally weighted blending weight
                     sum_p w(p) T_i(p) alpha_i(p) in the full-detail render,
                     read from the rasterizer's backward pass (the mass trick
                     from heatmap.py, on the real view). Off with --no-oracle

Metrics per image (only pixels the reference actually covers, alpha > 0.5):
  MSE/PSNR and SSIM over the full frame, the fovea (within --fovea deg of
  the gaze) and the periphery. LPIPS (AlexNet, inputs scaled to [-1, 1]) on
  the full frame and on a 64 x 64 patch centred on the gaze. Render time in
  ms from CUDA events.

Results are appended to a CSV after every frame, so an interrupted run can
continue with --resume.

Main run (truck: E1 budgets, E2 shuffled, E3 strength k, E5 live and oracle),
from the repo root with the clodgs env active:
    python gazeclod/eval_gaze_lod.py -m output/truck_clod --heat output/truck_clod/gaze_heat/heat_nthu_truck_density_sig2.5.npz --k-sweep 1 4 --tag e1
Summary of an existing CSV:
    python gazeclod/eval_gaze_lod.py -m output/truck_clod --summary output/truck_clod/gaze_lod/results_e1.csv
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import angle_map, make_minicam, project_dir  # noqa: E402
from gazeclod.common import load_model, lod_pipe, render_colors, render_rgb  # noqa: E402
from gazeclod.heatmap import DensitySampler, fixation_rows  # noqa: E402
from gazeclod.lod_policy import (LodState, heat_to_h, match_budget, s_vector,  # noqa: E402
                                 shuffled, user_number)
from lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
from utils.loss_utils import create_window  # noqa: E402

FIELDS = ["fold", "user", "row", "t", "budget", "s", "policy", "k", "s_base",
          "drawn_view", "raster", "drawn", "drawn_frac", "ms",
          "mse_full", "mse_fov", "mse_per", "ssim_full", "ssim_fov", "ssim_per",
          "lpips_full", "lpips_fov", "fov_px", "gaze_ecc"]
MIN_FOV_PX = 50
PATCH = 64


# ---------------------------------------------------------------- metrics
_WIN = {}


def ssim_map(a, b):
    """Per-pixel SSIM (same 11x11, sigma 1.5 window as 3DGS), (H, W)."""
    c = a.shape[0]
    if c not in _WIN:
        _WIN[c] = create_window(11, c).to(a.device).type_as(a)
    w = _WIN[c]
    a, b = a[None], b[None]
    mu1 = F.conv2d(a, w, padding=5, groups=c)
    mu2 = F.conv2d(b, w, padding=5, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=5, groups=c) - mu1_sq
    s2 = F.conv2d(b * b, w, padding=5, groups=c) - mu2_sq
    s12 = F.conv2d(a * b, w, padding=5, groups=c) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return m[0].mean(0)


def masked_mean(x, mask):
    if int(mask.sum()) < MIN_FOV_PX:
        return float("nan")
    return float(x[mask].mean())


def psnr(mse):
    return 10 * math.log10(1.0 / max(mse, 1e-10)) if mse == mse else float("nan")


def timed_render(cam, gs, pipe, bg, s):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    pkg = render_rgb(cam, gs, pipe, bg, s=s)
    end.record()
    torch.cuda.synchronize()
    return pkg, start.elapsed_time(end)


# ---------------------------------------------------------------- policies
class HeatBank:
    """
    All per-user heat of one file in memory, so any subset is a quick sum.
    A path like "a.npz+b.npz" adds the files user by user (e.g. gaze + head
    mass heat: every glance then credits both where the eyes and where the
    head pointed). Mass heat is normalized per glance, so the two are on the
    same scale.
    """

    def __init__(self, path):
        parts = path.split("+")
        z = np.load(parts[0])
        self.users = [str(u) for u in z["users"]]
        self.heat = z["heat"].astype(np.float32)
        for extra in parts[1:]:
            z2 = np.load(extra)
            if [str(u) for u in z2["users"]] != self.users:
                raise ValueError(f"{extra} has different users than {parts[0]}")
            self.heat = self.heat + z2["heat"]

    def h(self, users, gamma):
        idx = [self.users.index(u) for u in users]
        return heat_to_h(self.heat[idx].sum(0), gamma)


def build_policies(bank, heat_team, args, cross_bank, extra_banks=()):
    h_main = bank.h(heat_team, args.gamma)
    m = args.main_name
    pols = [("uniform", None, 0.0), (f"{m}_k{args.k:g}", h_main, args.k)]
    for kk in args.k_sweep:
        if kk != args.k:
            pols.append((f"{m}_k{kk:g}", h_main, kk))
    if not args.no_shuffled:
        pols.append((f"shuffled_k{args.k:g}", shuffled(h_main, seed=0), args.k))
    rng = np.random.default_rng(0)
    for n in args.people:
        if n >= len(heat_team):
            continue
        for r in range(args.repeats):
            subset = list(rng.choice(heat_team, n, replace=False))
            pols.append((f"gaze_n{n}_r{r}", bank.h(subset, args.gamma), args.k))
    for name, b in extra_banks:
        pols.append((f"{name}_k{args.k:g}", b.h(heat_team, args.gamma), args.k))
    if cross_bank is not None:
        pols.append((f"cross_k{args.k:g}", cross_bank.h(cross_bank.users, args.gamma), args.k))
    if not args.no_live:
        pols.append((f"live_k{args.k:g}", "live", args.k))
    if not args.no_oracle:
        pols.append((f"oracle_k{args.k:g}", "oracle", args.k))
    return pols


def fovea_mass(cam, gs, pipe, bg, wpix):
    """Per-Gaussian sum_p wpix(p) T_i(p) alpha_i(p) in the full-detail render, (N,) tensor."""
    n = gs.get_xyz.shape[0]
    colors = torch.full((n, 3), 0.5, device="cuda", requires_grad=True)
    pkg = render_colors(cam, gs, pipe, bg, colors, 1.0)
    (g,) = torch.autograd.grad((pkg["render"][0] * wpix).sum(), colors)
    return g[:, 0].detach()


# ---------------------------------------------------------------- summary
def summarize(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if not rows:
        print("empty CSV")
        return
    groups = defaultdict(list)
    for r in rows:
        groups[(float(r["s"]), r["policy"])].append(r)
    frames = len({(r["fold"], r["user"], r["row"]) for r in rows})
    print(f"\n{csv_path}: {frames} frames")
    print(f"{'s':>4} {'policy':<16} {'view k':>7} {'rast k':>7} {'all':>6} {'ms':>6} |"
          f" {'PSNR full':>9} {'fovea':>6} {'periph':>6} | {'SSIM fov':>8} | {'LPIPS full':>10} {'fov':>6} | {'fov win%':>8}")
    uni = {}
    for (s, pol), g in sorted(groups.items()):
        if pol == "uniform":
            uni[s] = {(r["fold"], r["user"], r["row"]): float(r["mse_fov"]) for r in g}

    def col(g, key):
        return np.array([float(r[key]) for r in g])

    for (s, pol), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "uniform", kv[0][1])):
        pf = [psnr(m) for m in col(g, "mse_full")]
        pv = [psnr(m) for m in col(g, "mse_fov")]
        pp = [psnr(m) for m in col(g, "mse_per")]
        wins = [float(r["mse_fov"]) < uni[s][(r["fold"], r["user"], r["row"])]
                for r in g if pol != "uniform" and float(r["mse_fov"]) == float(r["mse_fov"])
                and (r["fold"], r["user"], r["row"]) in uni.get(s, {})]
        win = f"{100 * np.mean(wins):7.0f}%" if wins else "       -"
        print(f"{s:4g} {pol:<16} {np.median(col(g, 'drawn_view')) / 1e3:7.1f} {np.median(col(g, 'raster')) / 1e3:7.1f}"
              f" {np.mean(col(g, 'drawn_frac')):6.1%} {np.median(col(g, 'ms')):6.2f} |"
              f" {np.nanmedian(pf):9.2f} {np.nanmedian(pv):6.2f} {np.nanmedian(pp):6.2f} |"
              f" {np.nanmedian(col(g, 'ssim_fov')):8.4f} |"
              f" {np.nanmedian(col(g, 'lpips_full')):10.4f} {np.nanmedian(col(g, 'lpips_fov')):6.4f} | {win}")
    print("Medians over frames. view k / rast k = thousands of Gaussians kept in view / actually rasterized,"
          " all = kept anywhere (CLoD-GS points_used) as a share of the model.")
    print("'fov win%' = share of frames where the policy's foveal error is below uniform's.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", help="NTHU heat .npz (per-user rows) from heatmap.py; 'a.npz+b.npz' sums files")
    ap.add_argument("--main-name", default="gaze", help="policy name prefix for the --heat file")
    ap.add_argument("--summary", help="only print the summary of this CSV and exit")
    ap.add_argument("--tag", default="e1", help="results file name: results_<tag>.csv")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--test-team", default="both", choices=["both", "even", "odd"])
    ap.add_argument("--s", type=float, nargs="+", default=[2, 4, 8, 12, 16], help="uniform scales = budgets")
    ap.add_argument("--budget", default="view", choices=["view", "global"],
                    help="match Gaussians in view (default) or CLoD-GS's global points_used")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--k-sweep", type=float, nargs="*", default=[])
    ap.add_argument("--people", type=int, nargs="*", default=[])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cross-heat", default=None, help="optional heat file from another site, all users")
    ap.add_argument("--extra-heat", nargs="*", default=[], metavar="NAME=PATH",
                    help="extra heat files of the same site, same team split (e.g. mass=...mass_sig2.5.npz)")
    ap.add_argument("--no-shuffled", action="store_true")
    ap.add_argument("--no-live", action="store_true", help="skip the one-glance live-gaze policy")
    ap.add_argument("--no-oracle", action="store_true", help="skip the foveal-content ceiling")
    ap.add_argument("--sigma", type=float, default=2.5, help="live-gaze heat falloff in degrees")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=0.5, help="test fixations per second per viewer")
    ap.add_argument("--vmax", type=float, default=100.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fovea", type=float, default=5.0)
    ap.add_argument("--max-users", type=int, default=0, help="debug: viewers per fold")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: frames per viewer, spread evenly")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summarize(args.summary)
        return
    if not args.heat:
        sys.exit("--heat is required")

    out_dir = os.path.join(args.model, "gaze_lod")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"results_{args.tag}.csv")
    done = set()
    if os.path.exists(out_csv):
        if not args.resume:
            sys.exit(f"{out_csv} exists. Use --resume to continue it or pick another --tag.")
        done = {(r["fold"], r["user"], r["row"]) for r in csv.DictReader(open(out_csv, newline=""))}
        print(f"Resuming: {len(done)} frames already done")
    new_file = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    bank = HeatBank(args.heat)
    cross_bank = HeatBank(args.cross_heat) if args.cross_heat else None
    extra_banks = [(spec.split("=", 1)[0], HeatBank(spec.split("=", 1)[1])) for spec in args.extra_heat]
    odd = [u for u in bank.users if user_number(u) % 2 == 1]
    even = [u for u in bank.users if user_number(u) % 2 == 0]
    folds = [("even", odd, even), ("odd", even, odd)]
    if args.test_team != "both":
        folds = [f for f in folds if f[0] == args.test_team]

    gs, _ = load_model(args.model, args.iteration)
    n_total = gs.get_xyz.shape[0]
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    ones = torch.ones((n_total, 3), device="cuda")
    lpips_net = LPIPS("alex", "0.1").cuda().eval()
    sampler = DensitySampler(gs, pipe, 48, args.sigma)
    W, fov = args.size, math.radians(args.fov)
    traces = {G.user_id(p): p for p in G.list_traces(args.eyenav, args.site, args.scene)}

    # warm up kernels so the first timings are not inflated
    tr0 = G.to_scene(G.load_trace(traces[bank.users[0]]))
    cam0 = make_minicam(tr0["R_c2w"][0], tr0["C"][0], W, W, fov, fov)
    for _ in range(3):
        timed_render(cam0, gs, pipe, bg, 1.0)

    t_start = time.time()
    for fold, heat_team, test_team in folds:
        pols = build_policies(bank, heat_team, args, cross_bank, extra_banks)
        print(f"\nFold '{fold}': heat from {len(heat_team)} people, testing {len(test_team)} viewers. "
              f"Policies: {', '.join(p[0] for p in pols)}")
        jobs = []
        for uid in test_team[: args.max_users or None]:
            tr = G.to_scene(G.load_trace(traces[uid]))
            rows, _ = fixation_rows(tr, args.fps, args.vmax)
            if args.max_frames and len(rows) > args.max_frames:
                rows = rows[np.linspace(0, len(rows) - 1, args.max_frames).astype(int)]  # spread over the session
            jobs += [(uid, tr, int(i)) for i in rows if (fold, uid, str(int(i))) not in done]
        for uid, tr, i in tqdm(jobs, desc=f"fold {fold}"):
            R, C, d = tr["R_c2w"][i], tr["C"][i], tr["gaze_d"][i]
            cam = make_minicam(R, C, W, W, fov, fov)
            st = LodState(gs, cam, budget=args.budget)
            with torch.no_grad():
                ref, _ = timed_render(cam, gs, pipe, bg, 1.0)
                alpha = render_colors(cam, gs, pipe, bg, ones, 1.0)["render"][0]
            content = alpha > 0.5
            ang = torch.tensor(angle_map(R, d, W, fov), device="cuda")
            m_fov, m_per = content & (ang < args.fovea), content & (ang >= args.fovea)
            uv = project_dir(R, d, W, W, fov, fov)
            patch = None
            if uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < W:
                x0 = int(min(max(round(uv[0]) - PATCH // 2, 0), W - PATCH))
                y0 = int(min(max(round(uv[1]) - PATCH // 2, 0), W - PATCH))
                patch = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
            ref_img = ref["render"]
            live_h = heat_to_h(sampler(R, C, d), args.gamma) if not args.no_live else None
            oracle_h = None
            if not args.no_oracle:
                wpix = torch.exp(-ang.float() ** 2 / (2 * args.sigma ** 2))
                oracle_h = heat_to_h(fovea_mass(cam, gs, pipe, bg, wpix), args.gamma)
            out_rows = []
            for s in args.s:
                target = st.count(s)
                for name, h, k in pols:
                    if isinstance(h, str):  # "live" or "oracle", computed per frame
                        h = live_h if h == "live" else oracle_h
                    if h is None:
                        sb, n, sv_flat, sv = s, target, s, s
                    else:
                        sb, n = match_budget(st, h, k, target, s)
                        sv_flat = s_vector(h, sb, k)
                        sv = sv_flat[:, None]
                    n_global = st.count_global(sv_flat)
                    with torch.no_grad():
                        pkg, ms = timed_render(cam, gs, pipe, bg, sv)
                        img = pkg["render"]
                        err = ((img - ref_img) ** 2).mean(0)
                        sm = ssim_map(img, ref_img)
                        lp_full = float(lpips_net(img[None] * 2 - 1, ref_img[None] * 2 - 1))
                        lp_fov = float("nan")
                        if patch is not None:
                            a = img[:, patch[0], patch[1]][None] * 2 - 1
                            b = ref_img[:, patch[0], patch[1]][None] * 2 - 1
                            lp_fov = float(lpips_net(a, b))
                    if pkg["points_used"] != n_global:
                        print(f"  warning: {name} kept {pkg['points_used']:,}, predicted {n_global:,}")
                    out_rows.append({
                        "fold": fold, "user": uid, "row": i, "t": round(tr["t_ms"][i] / 1000, 2),
                        "budget": args.budget, "s": s, "policy": name, "k": k, "s_base": round(float(sb), 4),
                        "drawn_view": n, "raster": int((pkg["radii"] > 0).sum()),
                        "drawn": pkg["points_used"], "drawn_frac": round(pkg["points_used"] / n_total, 5),
                        "ms": round(ms, 3),
                        "mse_full": masked_mean(err, content), "mse_fov": masked_mean(err, m_fov),
                        "mse_per": masked_mean(err, m_per),
                        "ssim_full": masked_mean(sm, content), "ssim_fov": masked_mean(sm, m_fov),
                        "ssim_per": masked_mean(sm, m_per),
                        "lpips_full": lp_full, "lpips_fov": lp_fov,
                        "fov_px": int(m_fov.sum()),
                        "gaze_ecc": round(float(np.degrees(np.arccos(np.clip(
                            np.dot(d / np.linalg.norm(d), R[:, 2]), -1, 1)))), 2),
                    })
            writer.writerows(out_rows)
            fh.flush()
    fh.close()
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Results: {out_csv}")
    summarize(out_csv)


if __name__ == "__main__":
    main()
