"""
Session 4: turn EyeNavGS gaze into a heat score for every Gaussian.

For each fixation sample we point a tiny camera straight down the gaze ray
(field of view +-3 sigma) and render it. Two ways to score a Gaussian:

  density (default): how close to the gaze point the Gaussian sits, if the eye
      could actually see it.
          heat_i += exp(-theta_i^2 / (2 sigma^2)) * visible_i
      theta_i is the angle between the gaze ray and the Gaussian's centre.
      visible_i compares the centre's depth with the rendered surface depth
      (alpha-normalized inverse depth from the same tiny render). Every
      Gaussian on a looked-at surface gets the same credit no matter its size,
      like a classic fixation density map, but on 3D blobs.

  mass: how much of what the fovea saw came from that Gaussian.
          heat_i += sum_p w(p) * T_i(p) * alpha_i(p),  w normalized to sum 1
      read from the rasterizer's backward pass (the image is linear in the
      per-Gaussian colors, so d(sum_p w C) / d c_i is exactly that sum).
      Exact, but it splits each glance among the Gaussians in view, so a few
      huge pavement blobs collect more than hundreds of tiny ones on the
      truck's logo. Kept as an ablation.

Saccades (gaze moving faster than --vmax deg/s) are dropped, since the eye
takes in almost nothing mid-jump. Every participant counts equally: a user's
heat is the mean over their samples. Each user's heat is stored separately so
later experiments can build a heatmap from any subset of people.

Run from the repo root with the clodgs env active:
    python gazeclod/heatmap.py -m output/truck_clod --scene truck --site rutgers
Writes <model>/gaze_heat/heat_<site>_<scene>_<mode>_sig<sigma>.npz
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import focal, make_minicam  # noqa: E402
from gazeclod.common import load_model, lod_pipe, render_colors  # noqa: E402


def fixation_rows(tr, hz=10.0, vmax=100.0, view=0):
    """
    Pick one gaze row per 1/hz seconds from one eye, skipping saccades.
    Speed = angle between consecutive gaze rays (world frame) / time step.
    Returns (row indices, fraction of that eye's rows that were saccades).
    """
    rows = np.where(tr["view"] == view)[0]
    if len(rows) < 2:
        return rows[:0], 0.0
    t = tr["t_ms"][rows] / 1000.0
    d = tr["gaze_d"][rows]
    ang = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1.0, 1.0)))
    dt = np.diff(t)
    speed = np.full(len(rows), np.inf)
    good = dt > 0
    speed[1:][good] = ang[good] / dt[good]
    ok = speed < vmax
    cand = np.where(ok)[0]
    bins = np.floor((t[cand] - t[0]) * hz).astype(np.int64)
    _, first = np.unique(bins, return_index=True)
    return rows[cand[first]], float((~ok[1:]).mean())


def gaze_camera_pose(R_c2w, d):
    """Camera-to-world rotation looking straight down gaze direction d, roll from the eye camera."""
    z = d / np.linalg.norm(d)
    x = np.cross(R_c2w[:, 1], z)  # eye camera's 'down' axis x gaze
    if np.linalg.norm(x) < 1e-6:
        x = R_c2w[:, 0].copy()
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def foveal_weights(px, fov, sigma_deg):
    """(px, px) weights exp(-theta^2 / 2 sigma^2) over pixel angle from the image centre, sum 1."""
    f = focal(fov, px)
    u = np.arange(px) - (px - 1) / 2.0  # 3DGS puts the optical axis at (S - 1) / 2
    X, Y = np.meshgrid(u, u)
    theta = np.arctan(np.sqrt(X ** 2 + Y ** 2) / f)
    w = np.exp(-theta ** 2 / (2 * math.radians(sigma_deg) ** 2))
    return torch.tensor(w / w.sum(), dtype=torch.float32, device="cuda")


class DensitySampler:
    """
    Density heat from ONE gaze sample: exp(-theta_i^2 / 2 sigma^2) * visible_i for
    every Gaussian, as an (N,) cuda tensor. Shared by the crowd heatmap (summed
    over many samples) and the live-gaze upper bound in eval_gaze_lod.py.
    """

    def __init__(self, gs, pipe, px, sigma_deg, tol=0.1, s=1.0):
        self.gs, self.pipe, self.px, self.tol, self.s = gs, pipe, px, tol, s
        self.fov = math.radians(6 * sigma_deg)  # gaze camera covers +-3 sigma
        self.xyz = gs.get_xyz.detach()
        n = self.xyz.shape[0]
        self.ones = torch.ones((n, 3), device="cuda")
        self.bg = torch.zeros(3, device="cuda")
        self.f = focal(self.fov, px)
        self.c0 = (px - 1) / 2.0
        self.cos_cone = math.cos(self.fov / 2)
        self.two_sig2 = 2 * math.radians(sigma_deg) ** 2
        self.inf = torch.full((px, px), float("inf"), device="cuda")

    @torch.no_grad()
    def __call__(self, R_c2w, C, d):
        px = self.px
        heat = torch.zeros(self.xyz.shape[0], device="cuda")
        R = gaze_camera_pose(R_c2w, d)
        cam = make_minicam(R, C, px, px, self.fov, self.fov)
        pkg = render_colors(cam, self.gs, self.pipe, self.bg, self.ones, self.s)
        acc = pkg["render"][0]   # accumulated opacity per pixel (colors are all 1, bg 0)
        inv = pkg["depth"][0]    # sum_i T_i alpha_i / z_i
        zs = torch.where(acc > 0.5, acc / inv.clamp_min(1e-8), self.inf)  # surface depth, inf = empty
        zs = F.max_pool2d(zs[None, None], 3, stride=1, padding=1)[0, 0]  # lenient at edges
        Rt = torch.tensor(R, dtype=torch.float32, device="cuda")
        Ct = torch.tensor(C, dtype=torch.float32, device="cuda")
        X = (self.xyz - Ct) @ Rt      # camera coords, +z along the gaze
        z = X[:, 2]
        cosang = z / X.norm(dim=1).clamp_min(1e-8)
        idx = ((z > 0.01) & (cosang > self.cos_cone)).nonzero().squeeze(1)
        if idx.numel() == 0:
            return heat
        Xi, zi = X[idx], z[idx]
        u = (self.f * Xi[:, 0] / zi + self.c0).round().long().clamp(0, px - 1)
        v = (self.f * Xi[:, 1] / zi + self.c0).round().long().clamp(0, px - 1)
        vis = zi <= zs[v, u] * (1 + self.tol)
        th = torch.acos(cosang[idx][vis].clamp(-1, 1))
        heat[idx[vis]] = torch.exp(-th ** 2 / self.two_sig2)
        return heat


def user_heat_density(gs, tr, rows, pipe, px, fov, sigma_deg, tol=0.1, s=1.0, desc=""):
    """Mean visible foveal weight at each Gaussian's centre over this user's samples, (N,) numpy."""
    sampler = DensitySampler(gs, pipe, px, sigma_deg, tol, s)
    heat = torch.zeros(gs.get_xyz.shape[0], device="cuda")
    for i in tqdm(rows, desc=desc, leave=False):
        heat += sampler(tr["R_c2w"][i], tr["C"][i], tr["gaze_d"][i])
    return (heat / max(len(rows), 1)).cpu().numpy()


def user_heat_mass(gs, tr, rows, pipe, wfov, px, fov, s=1.0, desc="", aim="gaze"):
    """
    Mean foveal blending weight per Gaussian over this user's samples, (N,) numpy.
    aim="gaze" points the tiny camera along the eye's gaze. aim="head" points it
    along the head's forward direction instead (control: no eye tracking).
    """
    aim_key = "gaze_d" if aim == "gaze" else "head_fwd"
    n = gs.get_xyz.shape[0]
    heat = torch.zeros(n, device="cuda")
    colors = torch.full((n, 3), 0.5, device="cuda", requires_grad=True)
    bg = torch.zeros(3, device="cuda")
    for i in tqdm(rows, desc=desc, leave=False):
        R = gaze_camera_pose(tr["R_c2w"][i], tr[aim_key][i])
        cam = make_minicam(R, tr["C"][i], px, px, fov, fov)
        pkg = render_colors(cam, gs, pipe, bg, colors, s)
        loss = (pkg["render"][0] * wfov).sum()
        (g,) = torch.autograd.grad(loss, colors)
        heat += g[:, 0]
    return (heat / max(len(rows), 1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="rutgers", choices=list(G.SITES))
    ap.add_argument("--sigma", type=float, default=2.5, help="foveal falloff in degrees")
    ap.add_argument("--hz", type=float, default=10.0, help="gaze samples per second per user")
    ap.add_argument("--vmax", type=float, default=100.0, help="saccade threshold, deg/s")
    ap.add_argument("--mode", default="density", choices=["density", "mass"])
    ap.add_argument("--tol", type=float, default=0.1, help="density mode: depth tolerance (0.1 = 10%%)")
    ap.add_argument("--px", type=int, default=48, help="size of the tiny gaze camera image")
    ap.add_argument("--aim", default="gaze", choices=["gaze", "head"],
                    help="mass mode control: point the camera along the head instead of the eyes")
    ap.add_argument("--window", default="fovea", choices=["fovea", "full"],
                    help="mass mode control: 'full' scores the whole 90 deg view with equal weight "
                         "(viewpoint significance, no gaze at all; use with --aim head)")
    ap.add_argument("--s", type=float, default=1.0, help="CLoD scale for the heat renders")
    ap.add_argument("--users", nargs="*", default=None, help="optional subset, e.g. user102 user103")
    args = ap.parse_args()

    if args.mode == "density" and (args.aim != "gaze" or args.window != "fovea"):
        sys.exit("--aim / --window controls are only implemented for --mode mass")
    fov = math.radians(6 * args.sigma)  # gaze camera covers +-3 sigma
    wfov = foveal_weights(args.px, fov, args.sigma)
    if args.window == "full":
        fov = math.radians(90.0)
        wfov = torch.full((args.px, args.px), 1.0 / args.px ** 2, device="cuda")
    gs, _ = load_model(args.model, args.iteration)
    pipe = lod_pipe()
    print(f"Loaded {gs.get_xyz.shape[0]:,} Gaussians. mode={args.mode}, sigma={args.sigma} deg, "
          f"{args.hz:g} Hz, saccades > {args.vmax:g} deg/s dropped")

    users, heats, nsamp = [], [], []
    t_start = time.time()
    for p in G.list_traces(args.eyenav, args.site, args.scene):
        uid = G.user_id(p)
        if args.users and uid not in args.users:
            continue
        tr = G.load_trace(p)
        if len(tr["view"]) == 0:
            print(f"  {uid}: no gaze recorded, skipped")
            continue
        tr = G.to_scene(tr)
        rows, sacc = fixation_rows(tr, args.hz, args.vmax)
        t0 = time.time()
        if args.mode == "density":
            h = user_heat_density(gs, tr, rows, pipe, args.px, fov, args.sigma, args.tol, args.s, desc=uid)
        else:
            h = user_heat_mass(gs, tr, rows, pipe, wfov, args.px, fov, args.s, desc=uid, aim=args.aim)
        users.append(uid)
        heats.append(h.astype(np.float32))
        nsamp.append(len(rows))
        print(f"  {uid}: {len(rows)} samples ({sacc:.0%} saccade rows dropped), "
              f"{(h > 0).mean():.1%} of Gaussians touched, {time.time() - t0:.0f} s")

    if not users:
        sys.exit("No usable users found.")
    out_dir = os.path.join(args.model, "gaze_heat")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.mode + ("" if args.aim == "gaze" else "_head") + ("" if args.window == "fovea" else "_full")
    out = os.path.join(out_dir, f"heat_{args.site}_{args.scene}_{tag}_sig{args.sigma:g}.npz")
    np.savez_compressed(out, users=np.array(users), heat=np.stack(heats), n_samples=np.array(nsamp),
                        mode=tag, aim=args.aim, window=args.window, sigma=args.sigma, hz=args.hz, vmax=args.vmax, px=args.px,
                        tol=args.tol, s=args.s)
    print(f"Saved {out}  ({len(users)} users, {sum(nsamp):,} samples, "
          f"{(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
