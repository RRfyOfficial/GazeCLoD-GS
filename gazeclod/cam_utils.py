"""
Small camera helpers shared by the GazeCLoD-GS scripts.

Poses come from gaze_io.to_scene(): R_c2w is a camera-to-world rotation in
COLMAP convention (x right, y down, z forward) and C is the camera centre.
That is exactly what 3DGS stores in Camera.R, with Camera.T = -R^T C.
"""
import math

import numpy as np


def focal(fov_rad, pixels):
    return pixels / (2.0 * math.tan(fov_rad / 2.0))


def project_dir(R_c2w, d, width, height, fovx, fovy):
    """
    Pixel (u, v) where a world-space direction d lands for a camera at the
    origin of that direction (fine for gaze: the gaze origin is ~1 cm from the
    eye). Returns None if the direction points behind the camera.
    """
    g = R_c2w.T @ d
    if g[2] <= 1e-6:
        return None
    u = focal(fovx, width) * g[0] / g[2] + width / 2.0
    v = focal(fovy, height) * g[1] / g[2] + height / 2.0
    return u, v


def make_minicam(R_c2w, C, width, height, fovx, fovy, znear=0.01, zfar=100.0):
    """A 3DGS MiniCam for an arbitrary pose (needs CUDA, imports the repo)."""
    import torch
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    R_c2w = np.asarray(R_c2w, dtype=np.float64)
    T = -R_c2w.T @ np.asarray(C, dtype=np.float64)
    wv = torch.tensor(getWorld2View2(R_c2w, T)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = MiniCam(width, height, fovy, fovx, znear, zfar, wv, full)
    # CLoD-GS's optional view-distance mode reads .R, and exposure lookup reads .image_name
    cam.R = R_c2w
    cam.T = T
    cam.image_name = "eyenavgs"
    return cam
