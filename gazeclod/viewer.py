"""
GazeCLoD-GS live viewer. Fly through the splat and compare plain CLoD-GS with
GazeCLoD-GS at the SAME number of Gaussians in view (same cost, so neither
side should feel faster; the question is which one looks better where you
look). Your mouse cursor plays the eye tracker: the red circle is your gaze.

Easiest way to start (Windows): double-click gazeclod/run_viewer.bat
Or from the repo root with the clodgs env active (close other GPU jobs first):
    python gazeclod/viewer.py                    (starts in truck)
    python gazeclod/viewer.py --scene playroom   (starts in playroom)
The viewer finds each scene's model and crowd heat files by itself. Needs an
NVIDIA GPU (the splat renderer is CUDA code), so it cannot run on a Mac.

Controls (click the window first; drag its corner to resize):
  hold W A S D      walk (SHIFT = faster), hold Q / E = down / up
  left drag         turn your head
  mouse             your eyes (red circle)
  [ ]               fewer blobs / more blobs (both sides always get the same number)
  1 2 3 4           what the RIGHT side shows: 1 ours, 2 ours + live eye tracker,
                    3 full detail (no limit), 4 the crowd heatmap
  Z                 zoom box on / off: a 3x close-up of what is inside the red circle, on both sides
  R                 replay a real recorded viewer (N = next viewer, SPACE = pause); R again to stop
  T                 switch scene (truck / playroom); each scene remembers where you were
  ESC               quit
  (extra: K cycles the strength k, M turns the quality readout off for more fps,
   P saves a screenshot to output/viewer_shots/)
"""
import argparse
import ctypes
import glob
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
from gazeclod.lod_policy import LodState, heat_to_h, s_vector, user_number  # noqa: E402
from gazeclod.show_heat import turbo  # noqa: E402

HELP = __doc__.split("Controls")[1].split("\n")[1:]
MODES = {1: "OURS: GazeCLoD-GS (crowd heatmap)", 2: "OURS + live eye tracker (your mouse)",
         3: "FULL DETAIL (no blob limit)", 4: "CROWD HEATMAP (red = looked at a lot)"}
LINE1 = "Move: hold W A S D   Turn: left-drag   Your eyes: the mouse (red circle)   Blobs: [ fewer   ] more"
LINE2 = "Right side shows: 1 ours   2 ours + eye tracker   3 full detail   4 heatmap      Z zoom   R replay a real viewer   T scene   ESC quit"


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


F_BIG, F_MED, F_SMALL = font(20), font(15), font(14)
WIN = "GazeCLoD-GS viewer"
S_MAX = 40.0  # past this, plain CLoD-GS hits its top-5000 floor and every image is unusable anyway

USER32 = ctypes.windll.user32 if os.name == "nt" else None


def held(vk):
    """True while a key is held (Windows). vk: virtual-key code, e.g. ord('W')."""
    return USER32 is not None and (USER32.GetAsyncKeyState(vk) & 0x8000) != 0


def window_focused():
    if USER32 is None:
        return True
    return USER32.GetForegroundWindow() == USER32.FindWindowW(None, WIN)


def rot(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


class Mouse:
    def __init__(self):
        self.pos, self.drag, self.dyaw, self.dpitch = None, None, 0.0, 0.0


def on_mouse(event, x, y, flags, m):
    if event == cv2.EVENT_MOUSEMOVE:
        m.pos = (x, y)
        if m.drag is not None:
            m.dyaw += (x - m.drag[0]) * 0.004
            m.dpitch += (y - m.drag[1]) * 0.004
            m.drag = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        m.drag = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        m.drag = None


def quick_match(lod, h, k, target, s, guess):
    """Budget matching warm-started from last frame's s_base (about 10 to 14 counts)."""
    lo, hi = s, (guess * 1.15 if guess else s * 2.0)
    if guess and guess / 1.15 > s and lod.count(s_vector(h, guess / 1.15, k)) > target:
        lo = guess / 1.15
    # Expand the bracket, but give up after a while: near CLoD-GS's top-5000 floor
    # an exact match may not exist (this used to loop forever when s got very large).
    for _ in range(25):
        if lod.count(s_vector(h, hi, k)) <= target:
            break
        lo, hi = hi, hi * 1.5
    else:
        return hi, lod.count(s_vector(h, hi, k))
    for _ in range(10):
        mid = math.sqrt(lo * hi)
        if lod.count(s_vector(h, mid, k)) > target:
            lo = mid
        else:
            hi = mid
    return hi, lod.count(s_vector(h, hi, k))


def timed(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    out = fn()
    b.record()
    torch.cuda.synchronize()
    return out, a.elapsed_time(b)


def fovea_psnr(img, ref, mask):
    if int(mask.sum()) < 50:
        return float("nan")
    err = ((img - ref) ** 2).mean(0)[mask]
    return 10 * math.log10(1.0 / max(float(err.mean()), 1e-10))


def draw_panel(img_chw, title, sub, uv, ring, accent, zoom=0, scale=3):
    im = Image.fromarray(to_uint8(img_chw))
    W = im.width
    d = ImageDraw.Draw(im)
    if zoom and uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < W:
        # 3x close-up of the gaze area (nearest-neighbour so you see the real pixels),
        # placed in the corner opposite the gaze so it never covers the circle
        u, v = uv
        x0 = int(min(max(round(u) - zoom // 2, 0), W - zoom))
        y0 = int(min(max(round(v) - zoom // 2, 0), W - zoom))
        inset = im.crop((x0, y0, x0 + zoom, y0 + zoom)).resize((zoom * scale, zoom * scale), Image.NEAREST)
        ix = 8 if u > W / 2 else W - inset.width - 8
        iy = 64 if v > W / 2 else W - inset.height - 8
        d.rectangle([x0, y0, x0 + zoom, y0 + zoom], outline=(255, 255, 255), width=1)
        im.paste(inset, (ix, iy))
        d.rectangle([ix - 3, iy - 3, ix + inset.width + 2, iy + inset.height + 2], outline=accent, width=3)
    if uv is not None:
        d.ellipse([uv[0] - ring, uv[1] - ring, uv[0] + ring, uv[1] + ring], outline=(255, 40, 40), width=3)
    d.rectangle([0, 0, im.width, 56], fill=(0, 0, 0))
    d.rectangle([0, 0, 6, 56], fill=accent)
    d.text((14, 4), title, fill=(255, 255, 255), font=F_BIG)
    d.text((14, 32), sub, fill=(210, 210, 210), font=F_MED)
    return im


def find_heat(model_dir, site, scene):
    """Best crowd heat for a scene: gaze+head (the headline combo), else gaze only, else stamp scoring."""
    def p(tag):
        return os.path.join(model_dir, "gaze_heat", f"heat_{site}_{scene}_{tag}_sig2.5.npz")
    for tags in (("mass", "mass_head"), ("mass",), ("density",)):
        paths = [p(t) for t in tags]
        if all(os.path.exists(x) for x in paths):
            return "+".join(paths)
    return None


def find_scenes(site):
    """Every output/<scene>_clod folder that has a trained model and a crowd heat file."""
    out = {}
    for d in sorted(glob.glob(os.path.join("output", "*_clod"))):
        name = os.path.basename(d)[:-len("_clod")]
        heat = find_heat(d, site, name)
        if heat and os.path.exists(os.path.join(d, "cameras.json")) and os.path.isdir(os.path.join(d, "point_cloud")):
            out[name] = (d, heat)
    return out


class Scene:
    """Everything the viewer needs for one scene, loaded once and kept on the GPU."""

    def __init__(self, name, model, heat, args):
        self.name = name
        self.bank = HeatBank(heat)
        self.gs, _ = load_model(model, args.iteration)
        self.h_crowd = self.bank.h(self.bank.users, args.gamma)
        gray = base_colors(self.gs).mean(1, keepdim=True).expand(-1, 3) * 0.5
        self.heat_colors = (1 - self.h_crowd[:, None]) * gray + self.h_crowd[:, None] * turbo(self.h_crowd)
        cams = json.load(open(os.path.join(model, "cameras.json")))
        Rs = np.array([c["rotation"] for c in cams])
        up = -Rs[:, :, 1].mean(0)
        self.up = up / np.linalg.norm(up)
        self.R = np.array(cams[0]["rotation"], dtype=np.float64)  # where you stand when you first enter
        self.C = np.array(cams[0]["position"], dtype=np.float64)
        xyz = self.gs.get_xyz.detach()
        self.speed = 0.3 * float(torch.norm(xyz - xyz.mean(0), dim=1).mean())  # scene units per second
        try:
            self.traces = G.list_traces(args.eyenav, args.site, name)
        except (KeyError, OSError):
            self.traces = []


def message_frame(W, text, sub=""):
    frame = Image.new("RGB", (2 * W + 6, W + 54), (0, 0, 0))
    d = ImageDraw.Draw(frame)
    d.text((40, W // 2 - 20), text, fill=(255, 255, 255), font=F_BIG)
    d.text((40, W // 2 + 12), sub, fill=(170, 170, 170), font=F_MED)
    cv2.imshow(WIN, cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description="GazeCLoD-GS live viewer. With no arguments it opens truck; press T to switch scenes.")
    ap.add_argument("--scene", default="truck", help="scene to start in (truck or playroom)")
    ap.add_argument("-m", "--model", default=None, help="optional: model folder (default: output/<scene>_clod)")
    ap.add_argument("--heat", default=None, help="optional: heat .npz, 'a.npz+b.npz' sums files (default: found automatically)")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--eyenav", default=None, help="EyeNavGS folder (default: data/eyenavgs)")
    ap.add_argument("--site", default="nthu")
    ap.add_argument("--s", type=float, default=12.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=640, help="render size per panel (window can be resized)")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fovea", type=float, default=5.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("This viewer needs an NVIDIA GPU with CUDA (the splat renderer is CUDA code), so it can't run here.\n"
              "On a Mac, watch the demo video instead: output/truck_clod/gaze_lod/video_user12_s16_k4_h264.mp4")
        sys.exit(1)

    # paths you typed are relative to where you ran the command; everything else is relative to the repo
    if args.model:
        args.model = os.path.abspath(args.model)
    if args.heat:
        args.heat = "+".join(os.path.abspath(x) for x in args.heat.split("+"))
    args.eyenav = os.path.abspath(args.eyenav) if args.eyenav else os.path.join(REPO, "data", "eyenavgs")
    os.chdir(REPO)

    paths = find_scenes(args.site)
    if args.model:
        heat = args.heat or find_heat(args.model, args.site, args.scene)
        if heat is None:
            sys.exit(f"No heat file found in {args.model}/gaze_heat. Pass one with --heat.")
        paths[args.scene] = (args.model, heat)
    if args.scene not in paths:
        want = os.path.join("output", f"{args.scene}_clod")
        sys.exit(f"Scene '{args.scene}' is not ready. Scenes found: {', '.join(paths) or 'none'}.\n"
                 f"It needs a trained model in {want} and a heat file. Make the heat with:\n"
                 f"    python gazeclod/heatmap.py -m {want} --scene {args.scene} --site {args.site} --mode mass")
    names = list(paths)
    print(__doc__.split("Controls")[0] + "Controls" + "\n".join(HELP))
    print(f"Scenes found: {', '.join(names)} (press T to switch)")
    if USER32 is None:
        print("Note: smooth held-key movement needs Windows. Elsewhere each key press moves one step.")

    W, fov = args.size, math.radians(args.fov)
    f, c0 = focal(fov, W), (W - 1) / 2.0
    ring = f * math.tan(math.radians(args.fovea))
    pipe = lod_pipe()
    bg = torch.zeros(3, device="cuda")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 2 * W + 6, W + 54)
    mouse = Mouse()
    cv2.setMouseCallback(WIN, on_mouse, mouse)

    loaded = {}

    def get_scene(name):
        if name not in loaded:
            message_frame(W, f"Loading {name} ...", "first time only, takes a few seconds; after that T switches instantly")
            print(f"Loading {name} from {paths[name][0]}")
            try:
                loaded[name] = Scene(name, *paths[name], args)
            except RuntimeError as e:  # GPU full: drop the other scenes and try once more
                if "out of memory" not in str(e).lower():
                    raise
                loaded.clear()
                torch.cuda.empty_cache()
                loaded[name] = Scene(name, *paths[name], args)
        return loaded[name]

    sc = get_scene(args.scene)
    R, C, speed = sc.R, sc.C, sc.speed
    h_active = sc.h_crowd
    replay_idx, replay, paused = 0, None, False
    mode, s, k = 1, args.s, args.k
    show_metric, frame_no, quality, zoom_on = True, 0, {}, False
    sb_cache = {}
    note, note_until = "", 0.0
    t_prev, fps = time.time(), 0.0

    while True:
        now = time.time()
        dt, t_prev = min(now - t_prev, 0.25), now
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        focused = window_focused()
        gs, up = sc.gs, sc.up

        # ---------------- pose
        if replay is not None:
            tr, rows, t_rel, play_t, wall0 = replay
            if paused:
                wall0 = now - play_t
            else:
                play_t = now - wall0
            replay = (tr, rows, t_rel, play_t, wall0)
            j = min(int(np.searchsorted(t_rel, play_t)), len(rows) - 1)
            i = rows[j]
            R, C, gaze_d = tr["R_c2w"][i].copy(), tr["C"][i].copy(), tr["gaze_d"][i]
            if j == len(rows) - 1:
                replay, h_active = None, sc.h_crowd
        else:
            if mouse.dyaw or mouse.dpitch:
                R = orthonormalize(rot(up, -mouse.dyaw) @ R @ rot(np.array([1.0, 0, 0]), -mouse.dpitch))
                mouse.dyaw = mouse.dpitch = 0.0
            if focused and USER32 is not None:
                v = speed * dt * (3.0 if held(0x10) else 1.0)  # 0x10 = SHIFT
                C = C + v * (held(ord("W")) - held(ord("S"))) * R[:, 2] \
                      + v * (held(ord("D")) - held(ord("A"))) * R[:, 0] \
                      + v * (held(ord("E")) - held(ord("Q"))) * up
            mx, my = mouse.pos if mouse.pos else (W // 2, W // 2)
            u = mx - (W + 6) if mx >= W + 6 else mx
            u, my = min(max(u, 0), W - 1), min(max(my, 0), W - 1)
            d_cam = np.array([(u - c0) / f, (my - c0) / f, 1.0])
            gaze_d = R @ (d_cam / np.linalg.norm(d_cam))

        cam = make_minicam(R, C, W, W, fov, fov)
        lod = LodState(gs, cam)
        target = lod.count(s)
        g = R.T @ gaze_d
        uv = (f * g[0] / g[2] + c0, f * g[1] / g[2] + c0) if g[2] > 0 else None
        ang = None

        def render_method():
            nonlocal ang
            if mode in (1, 2):
                h = h_active
                if mode == 2:
                    ang = torch.tensor(angle_map(R, gaze_d, W, fov), device="cuda", dtype=torch.float32)
                    h = heat_to_h(fovea_mass(cam, gs, pipe, bg, torch.exp(-ang ** 2 / (2 * 2.5 ** 2))), args.gamma)
                key = (mode, k, s)
                sb, n = quick_match(lod, h, k, target, s, sb_cache.get(key))
                sb_cache[key] = sb
                with torch.no_grad():
                    img, ms = timed(lambda: render_rgb(cam, gs, pipe, bg, s=s_vector(h, sb, k)[:, None])["render"])
                return img, ms, n
            with torch.no_grad():
                if mode == 3:
                    img, ms = timed(lambda: render_rgb(cam, gs, pipe, bg, s=1.0)["render"])
                    return img, ms, lod.count(1.0)
                img, ms = timed(lambda: render_colors(cam, gs, pipe, bg, sc.heat_colors, 1.0)["render"])
                return img, ms, None

        def render_plain():
            with torch.no_grad():
                img, ms = timed(lambda: render_rgb(cam, gs, pipe, bg, s=s)["render"])
            return img, ms, target

        panels = [("PLAIN CLoD-GS", *render_plain()), (MODES[mode], *render_method())]

        # quality where you look, vs full detail, refreshed every 4th frame (one extra render)
        frame_no += 1
        if show_metric and mode in (1, 2):
            if frame_no % 4 == 1 or not quality:
                with torch.no_grad():
                    ref = render_rgb(cam, gs, pipe, bg, s=1.0)["render"]
                if ang is None:
                    ang = torch.tensor(angle_map(R, gaze_d, W, fov), device="cuda", dtype=torch.float32)
                m = ang < args.fovea
                quality = {c: fovea_psnr(p[1], ref, m) for c, p in enumerate(panels)}
        else:
            quality = {}

        # ---------------- compose: left = plain (orange bar), right = chosen method (blue bar)
        frame = Image.new("RGB", (2 * W + 6, W + 54), (0, 0, 0))
        accents = [(244, 114, 48), (37, 125, 224)]
        for c, (title, img, ms, n) in enumerate(panels):
            parts = [f"{n:,} blobs"] if n is not None else []
            parts.append(f"{ms:.0f} ms")
            q = quality.get(c)
            if q is not None and q == q:
                parts.append(f"quality where you look: {q:.1f} dB")
            show_ring = uv is not None and not (c == 1 and mode == 4)
            frame.paste(draw_panel(img, title, "    ".join(parts), uv if show_ring else None, ring, accents[c],
                                   zoom=(80 if zoom_on and show_ring else 0)), (c * (W + 6), 0))
        dr = ImageDraw.Draw(frame)
        status = ""
        if replay is not None:
            status = f"   |   REPLAY of {G.user_id(sc.traces[replay_idx])}" + (" (paused)" if paused else "")
        dr.text((10, W + 6), LINE1 + status, fill=(235, 235, 235), font=F_SMALL)
        dr.text((10, W + 28), LINE2 + f"      {fps:.0f} fps", fill=(170, 170, 170), font=F_SMALL)
        label = f"Scene: {sc.name.upper()}" + ("  (T switches)" if len(names) > 1 else "")
        try:
            lw = dr.textlength(label, font=F_MED)
        except AttributeError:
            lw = 9 * len(label)
        dr.rectangle([2 * W + 6 - lw - 22, W + 4, 2 * W + 6, W + 26], fill=(0, 0, 0))
        dr.text((2 * W + 6 - lw - 12, W + 5), label, fill=(255, 210, 90), font=F_MED)
        if note and now < note_until:  # short message box, top middle
            try:
                nw = dr.textlength(note, font=F_MED)
            except AttributeError:
                nw = 9 * len(note)
            x = (2 * W + 6 - nw) / 2
            dr.rectangle([x - 14, 66, x + nw + 14, 96], fill=(0, 0, 0), outline=(255, 210, 90), width=2)
            dr.text((x, 72), note, fill=(255, 255, 255), font=F_MED)
        cv2.imshow(WIN, cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))

        # ---------------- discrete keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if USER32 is None and replay is None:  # fallback: one step per key press
            v = speed * 0.1
            C = C + v * ((key == ord("w")) - (key == ord("s"))) * R[:, 2] + v * ((key == ord("d")) - (key == ord("a"))) * R[:, 0]
        if key == ord("-"): speed /= 1.5
        if key == ord("="): speed *= 1.5
        if key == ord("["): s = min(S_MAX, s * 1.25)
        if key == ord("]"): s = max(1.0, s / 1.25)
        if key in (ord("1"), ord("2"), ord("3"), ord("4")): mode = key - ord("0")
        if key == ord("k"): k = {1.0: 2.0, 2.0: 4.0, 4.0: 1.0}.get(k, 2.0)
        if key == ord("m"): show_metric = not show_metric
        if key == ord("z"): zoom_on = not zoom_on
        if key == ord("p"):  # screenshot of exactly what is on screen, at full render resolution
            os.makedirs(os.path.join("output", "viewer_shots"), exist_ok=True)
            shot = os.path.join("output", "viewer_shots", f"{sc.name}_{time.strftime('%Y%m%d_%H%M%S')}.png")
            frame.save(shot)
            note, note_until = f"screenshot saved: {shot}", now + 3
            print(f"Screenshot saved: {os.path.abspath(shot)}")
        if key == ord(" "): paused = not paused
        if key in (ord("t"), ord("T")):
            if len(names) < 2:
                note, note_until = "only one scene is ready (make the other scene's heat file first)", now + 4
            else:
                if replay is None:  # remember where you were in this scene
                    sc.R, sc.C = R, C
                sc.speed = speed
                sc = get_scene(names[(names.index(sc.name) + 1) % len(names)])
                R, C, speed = sc.R, sc.C, sc.speed
                replay, paused, replay_idx, h_active = None, False, 0, sc.h_crowd
                sb_cache, quality = {}, {}
                mouse.dyaw = mouse.dpitch = 0.0
                t_prev = time.time()  # loading time should not count as one giant step
                note, note_until = f"now in {sc.name}", time.time() + 2
        if key in (ord("r"), ord("n")):
            if not sc.traces:
                note, note_until = f"no recorded viewers found for {sc.name} in {args.eyenav}", now + 4
            else:
                if key == ord("n"):
                    replay_idx = (replay_idx + 1) % len(sc.traces)
                if key == ord("r") and replay is not None:
                    replay, h_active = None, sc.h_crowd
                else:
                    tr = G.to_scene(G.load_trace(sc.traces[replay_idx]))
                    rows = np.where(tr["view"] == 0)[0]
                    t_rel = (tr["t_ms"][rows] - tr["t_ms"][rows][0]) / 1000.0
                    replay, paused = (tr, rows, t_rel, 0.0, time.time()), False
                    me = G.user_id(sc.traces[replay_idx])
                    others = [u for u in sc.bank.users if user_number(u) % 2 != user_number(me) % 2]
                    h_active = sc.bank.h(others, args.gamma)
                    print(f"Replaying {me} in {sc.name}, crowd heat from the other team ({len(others)} people)")
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
