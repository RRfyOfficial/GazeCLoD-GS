"""
GazeCLoD-GS live viewer. Fly through the splat and compare plain CLoD-GS
(left) with another method (right) at the SAME number of Gaussians in view.
Your mouse cursor plays the eye tracker: the red circle is "where you look".

Run from the repo root with the clodgs env active (close other GPU jobs first):
    python gazeclod/viewer.py -m output/truck_clod --scene truck --heat output/truck_clod/gaze_heat/heat_nthu_truck_mass_sig2.5.npz+output/truck_clod/gaze_heat/heat_nthu_truck_mass_head_sig2.5.npz

Controls (click the window first so it has keyboard focus):
  mouse move        your gaze (red circle)
  left drag         look around
  W A S D           move forward / left / back / right
  Q E               move down / up
  - =               slower / faster movement
  [ ]               tighter / looser budget (uniform scale s)
  1                 right panel: GazeCLoD-GS, crowd heat (what the experiments test)
  2                 right panel: live gaze, protects what is under YOUR cursor (needs an eye tracker in real life)
  3                 right panel: full detail (s = 1, no budget)
  4                 right panel: the crowd heatmap itself (blue = ignored, red = looked at)
  K                 cycle strength k (1, 2, 4)
  R                 replay a recorded NTHU viewer (their head pose and gaze, heat from the other team); R again to stop
  N                 next recorded viewer
  SPACE             pause / resume replay
  M                 fovea PSNR readout on / off (costs one extra render)
  H                 help on / off
  ESC               quit
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gazeclod import gaze_io as G  # noqa: E402
from gazeclod.cam_utils import angle_map, focal, make_minicam  # noqa: E402
from gazeclod.common import base_colors, load_model, lod_pipe, render_colors, render_rgb, to_uint8  # noqa: E402
from gazeclod.eval_gaze_lod import HeatBank, fovea_mass  # noqa: E402
from gazeclod.lod_policy import LodState, heat_to_h, match_budget, s_vector  # noqa: E402
from gazeclod.show_heat import turbo  # noqa: E402

HELP = __doc__.split("Controls")[1].split("\n")[1:]
MODES = {1: "GazeCLoD-GS (crowd heat)", 2: "live gaze (your cursor)", 3: "full detail (s = 1)", 4: "crowd heatmap"}


def rot(axis, ang):
    """Rotation matrix about a unit axis (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


class State:
    def __init__(self):
        self.mouse = None
        self.drag = None
        self.dyaw = 0.0
        self.dpitch = 0.0


def on_mouse(event, x, y, flags, st):
    if event == cv2.EVENT_MOUSEMOVE:
        st.mouse = (x, y)
        if st.drag is not None:
            st.dyaw += (x - st.drag[0]) * 0.004
            st.dpitch += (y - st.drag[1]) * 0.004
            st.drag = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        st.drag = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        st.drag = None


def fovea_psnr(img, ref, mask):
    if int(mask.sum()) < 50:
        return float("nan")
    err = ((img - ref) ** 2).mean(0)[mask]
    return 10 * math.log10(1.0 / max(float(err.mean()), 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--heat", required=True, help="heat .npz, 'a.npz+b.npz' sums files")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default="data/eyenavgs")
    ap.add_argument("--scene", default="truck")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--s", type=float, default=12.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fovea", type=float, default=5.0)
    args = ap.parse_args()

    print(__doc__.split("Controls")[0])
    print("Controls" + "\n".join(HELP))
    bank = HeatBank(args.heat)
    gs, _ = load_model(args.model, args.iteration)
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")
    h_crowd = bank.h(bank.users, args.gamma)  # all recorded viewers (free flight)
    h_active = h_crowd
    gray = base_colors(gs).mean(1, keepdim=True).expand(-1, 3) * 0.5
    hv = h_crowd.clamp(0, 1)
    heat_colors = (1 - hv[:, None]) * gray + hv[:, None] * turbo(hv)
    W, fov = args.size, math.radians(args.fov)
    f = focal(fov, W)
    c0 = (W - 1) / 2.0
    ring = f * math.tan(math.radians(args.fovea))

    # start at the first photo camera, up = average photo "up" (COLMAP y points down)
    cams = json.load(open(os.path.join(args.model, "cameras.json")))
    Rs = np.array([c["rotation"] for c in cams])
    up = -Rs[:, :, 1].mean(0)  # column 1 of camera-to-world = camera "down" axis in world
    up /= np.linalg.norm(up)
    R = np.array(cams[0]["rotation"], dtype=np.float64)
    C = np.array(cams[0]["position"], dtype=np.float64)
    xyz = gs.get_xyz.detach()
    scene_scale = float(torch.norm(xyz - xyz.mean(0), dim=1).mean())
    step = 0.02 * scene_scale

    traces = G.list_traces(args.eyenav, args.site, args.scene)
    replay_idx, replay, paused = 0, None, False
    mode, s, k = 1, args.s, args.k
    show_help, show_metric = True, True
    last_sb = None
    win = "GazeCLoD-GS viewer"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    st = State()
    cv2.setMouseCallback(win, on_mouse, st)
    t_prev = time.time()

    while True:
        # --- pose: replay or free flight
        if replay is not None:
            tr, rows, t_rel, play_t, wall0 = replay
            if not paused:
                play_t = (time.time() - wall0)
            j = min(int(np.searchsorted(t_rel, play_t)), len(rows) - 1)
            replay = (tr, rows, t_rel, play_t, wall0 if not paused else time.time() - play_t)
            i = rows[j]
            R, C, gaze_d = tr["R_c2w"][i].copy(), tr["C"][i].copy(), tr["gaze_d"][i]
            if j == len(rows) - 1:
                replay, h_active = None, h_crowd
        else:
            if st.dyaw or st.dpitch:
                R = rot(up, -st.dyaw) @ R  # drag right turns right
                R = R @ rot(np.array([1.0, 0, 0]), -st.dpitch)
                R = orthonormalize(R)
                st.dyaw = st.dpitch = 0.0
            mx, my = st.mouse if st.mouse else (W // 2, W // 2)
            u = (mx % (W + 6)) if mx < 2 * W + 6 else W // 2
            u = min(u, W - 1)
            d_cam = np.array([(u - c0) / f, (my - c0) / f, 1.0])
            gaze_d = R @ (d_cam / np.linalg.norm(d_cam))

        cam = make_minicam(R, C, W, W, fov, fov)
        lod = LodState(gs, cam)
        target = lod.count(s)
        ang = torch.tensor(angle_map(R, gaze_d, W, fov), device="cuda", dtype=torch.float32)
        g = R.T @ gaze_d
        uv = (f * g[0] / g[2] + c0, f * g[1] / g[2] + c0) if g[2] > 0 else None

        with torch.no_grad():
            left = render_rgb(cam, gs, pipe, bg, s=s)["render"]
        right_label, n_right = MODES[mode], None
        if mode == 1:
            sb, n_right = match_budget(lod, h_active, k, target, s, iters=20)
            with torch.no_grad():
                right = render_rgb(cam, gs, pipe, bg, s=s_vector(h_active, sb, k)[:, None])["render"]
        elif mode == 2:
            wpix = torch.exp(-ang ** 2 / (2 * 2.5 ** 2))
            h_live = heat_to_h(fovea_mass(cam, gs, pipe, bg, wpix), args.gamma)
            sb, n_right = match_budget(lod, h_live, k, target, s, iters=20)
            with torch.no_grad():
                right = render_rgb(cam, gs, pipe, bg, s=s_vector(h_live, sb, k)[:, None])["render"]
        elif mode == 3:
            with torch.no_grad():
                right = render_rgb(cam, gs, pipe, bg, s=1.0)["render"]
            n_right = lod.count(1.0)
        else:
            with torch.no_grad():
                right = render_colors(cam, gs, pipe, bg, heat_colors, 1.0)["render"]

        psnr_l = psnr_r = float("nan")
        if show_metric and mode in (1, 2):
            with torch.no_grad():
                ref = render_rgb(cam, gs, pipe, bg, s=1.0)["render"]
            m = ang < args.fovea
            psnr_l, psnr_r = fovea_psnr(left, ref, m), fovea_psnr(right, ref, m)

        # --- compose frame
        now = time.time()
        fps, t_prev = 1.0 / max(now - t_prev, 1e-6), now
        frame = Image.new("RGB", (2 * W + 6, W + 30), (0, 0, 0))
        for x0, img, title, n, p in [(0, left, f"CLoD-GS uniform, s = {s:g}", target, psnr_l),
                                     (W + 6, right, right_label + (f", k = {k:g}" if mode in (1, 2) else ""), n_right, psnr_r)]:
            im = Image.fromarray(to_uint8(img))
            d = ImageDraw.Draw(im)
            if uv is not None and (x0 == 0 or mode != 4):
                d.ellipse([uv[0] - ring, uv[1] - ring, uv[0] + ring, uv[1] + ring], outline=(255, 40, 40), width=2)
            d.rectangle([0, 0, W, 38], fill=(0, 0, 0))
            d.text((8, 4), title, fill=(255, 255, 255))
            sub = (f"{n:,} Gaussians in view" if n is not None else "")
            if p == p:
                sub += f"  |  fovea PSNR {p:.1f} dB"
            d.text((8, 21), sub, fill=(200, 200, 200))
            frame.paste(im, (x0, 0))
        dr = ImageDraw.Draw(frame)
        status = (f"{fps:4.1f} fps | " + (f"REPLAY {G.user_id(traces[replay_idx])}" + (" (paused)" if paused else "")
                  if replay is not None else "free flight (mouse = your gaze)") + " | H for help")
        dr.text((8, W + 9), status, fill=(220, 220, 220))
        if show_help:
            dr.rectangle([12, 48, 470, 48 + 15 * len(HELP)], fill=(0, 0, 0))
            for li, line in enumerate(HELP):
                dr.text((18, 50 + 15 * li), line.strip(), fill=(230, 230, 230))
        cv2.imshow(win, cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))

        # --- keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        fwd, right_v = R[:, 2], R[:, 0]
        if replay is None:
            if key == ord("w"): C = C + step * fwd
            if key == ord("s"): C = C - step * fwd
            if key == ord("a"): C = C - step * right_v
            if key == ord("d"): C = C + step * right_v
            if key == ord("e"): C = C + step * up
            if key == ord("q"): C = C - step * up
        if key == ord("-"): step /= 1.5
        if key == ord("="): step *= 1.5
        if key == ord("["): s = max(1.0, s * 1.25)
        if key == ord("]"): s = max(1.0, s / 1.25)
        if key in (ord("1"), ord("2"), ord("3"), ord("4")): mode = key - ord("0")
        if key == ord("k"): k = {1.0: 2.0, 2.0: 4.0, 4.0: 1.0}.get(k, 2.0)
        if key == ord("m"): show_metric = not show_metric
        if key == ord("h"): show_help = not show_help
        if key == ord(" "): paused = not paused
        if key in (ord("r"), ord("n")):
            if key == ord("n"):
                replay_idx = (replay_idx + 1) % len(traces)
            if key == ord("r") and replay is not None:
                replay, h_active = None, h_crowd
            else:
                tr = G.to_scene(G.load_trace(traces[replay_idx]))
                rows = np.where(tr["view"] == 0)[0]
                t_rel = (tr["t_ms"][rows] - tr["t_ms"][rows][0]) / 1000.0
                replay, paused = (tr, rows, t_rel, 0.0, time.time()), False
                # honest replay: the crowd heat comes only from the OTHER team, as in the experiments
                me = G.user_id(traces[replay_idx])
                par = int("".join(c for c in me if c.isdigit())) % 2
                others = [u for u in bank.users if int("".join(c for c in u if c.isdigit())) % 2 != par]
                h_active = bank.h(others, args.gamma)
                print(f"Replaying {me}, crowd heat from the other team ({len(others)} people)")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
