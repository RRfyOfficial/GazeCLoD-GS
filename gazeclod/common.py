"""
Shared helpers for loading the trained CLoD-GS model and rendering it.
Everything here needs CUDA and the compiled rasterizer.
"""
import json
import math
import os
import sys
from argparse import Namespace  # noqa: F401  (needed to eval cfg_args)

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gaussian_renderer import render  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from gazeclod.cam_utils import make_minicam  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True
except Exception:
    SEPARATE_SH = False

SH_C0 = 0.28209479177387814


def load_model(model_dir, iteration=30000):
    """Load the trained splat with every parameter frozen (we never train here)."""
    with open(os.path.join(model_dir, "cfg_args")) as f:
        cfg = eval(f.read())  # 3DGS saves its args as a Namespace(...) string
    gs = GaussianModel(cfg.sh_degree, enable_opacity_decay_train=True)
    gs.load_ply(os.path.join(model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation",
                 "_opacity", "_opacity_dis_decay"]:
        t = getattr(gs, name, None)
        if isinstance(t, torch.Tensor):
            t.requires_grad_(False)
    return gs, cfg


def lod_pipe():
    """The switches render.py uses: decayed opacity plus distance culling."""
    return Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False,
                     antialiasing=False, enable_decay_opacity_render=True,
                     enable_distance_culling=True, opacity_threshold=0.005,
                     enable_view_distance=False)


def render_rgb(cam, gs, pipe, bg, s=1.0):
    return render(cam, gs, pipe, bg, virtual_scale=s, separate_sh=SEPARATE_SH)


def render_colors(cam, gs, pipe, bg, colors, s=1.0):
    """
    Render with one fixed RGB per Gaussian. colors is (N, 3) for ALL Gaussians.

    CLoD-GS's render() culls means/opacities/scales with its LOD mask but passes
    override_color through untouched, so we compute the same mask here and
    hand it only the surviving rows.
    """
    mask, _ = gs.get_lod_filter_mask(cam, pipe.enable_distance_culling,
                                     pipe.enable_view_distance, virtual_depth_scale=s,
                                     opacity_threshold=pipe.opacity_threshold)
    c = colors if mask is None else colors[mask]
    pkg = render(cam, gs, pipe, bg, override_color=c, virtual_scale=s, separate_sh=False)
    if mask is not None and pkg["points_used"] != int(mask.sum()):
        raise RuntimeError("LOD mask mismatch between render_colors and render()")
    return pkg


def base_colors(gs):
    """Each Gaussian's view-independent color (the SH DC term), (N, 3) in [0, 1]."""
    return (0.5 + SH_C0 * gs._features_dc[:, 0, :]).clamp(0, 1)


def photo_cameras(model_dir, width=800, every=None, ids=None):
    """
    MiniCams for the original photo viewpoints (from cameras.json, which 3DGS
    writes with camera-to-world rotation and centre). Resized to `width`.
    """
    with open(os.path.join(model_dir, "cameras.json")) as f:
        cams = json.load(f)
    if ids is not None:
        cams = [c for c in cams if c["id"] in ids]
    elif every:
        cams = cams[::every]
    out = []
    for c in cams:
        fovx = 2 * math.atan(c["width"] / (2 * c["fx"]))
        fovy = 2 * math.atan(c["height"] / (2 * c["fy"]))
        h = int(round(width * c["height"] / c["width"]))
        cam = make_minicam(np.array(c["rotation"]), np.array(c["position"]), width, h, fovx, fovy)
        cam.image_name = c["img_name"]
        out.append(cam)
    return out


def to_uint8(img_chw):
    return (img_chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
