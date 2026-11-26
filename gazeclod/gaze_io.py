"""
Load EyeNavGS eye-tracking traces and convert them into the 3DGS / COLMAP
world frame of the trained splat.

How the EyeNavGS recorder works (read from their SIBR fork,
src/core/openxr/OpenXRRdrMode.cpp and OpenXRHMD.cpp):

  1. The per-scene settings (initial quaternion, scale, initial position) are
     applied inside OpenXR *before* anything is written to the CSV. So the
     Position / Quaternion / GazePos / GazeQ columns are already in the
     "render space" the viewer used. We never apply scene_setting.csv again.
  2. Right before rendering, the viewer flips 180 degrees about X
     (diag(1,-1,-1)) because OpenXR is +y up / +z backward and 3DGS scenes are
     +y down / +z forward. Replay mode feeds the CSV pose through the same
     path, so this flip is the whole mapping from CSV to splat world.
  3. Gaze rows come from XR_FB_eye_tracking_social in the same play space.
     The gaze direction is the -Z axis of the gaze quaternion (checked on the
     data: about 12-19 deg from head-forward, vs about 165 deg for +Z).

Column order differs between sites (Rutgers writes GazePos before GazeQ,
NTHU the other way round), so everything is read by column name.
"""
import csv
import glob
import os

import numpy as np

# OpenXR (x right, y up, z back)  ->  3DGS world (x right, y down, z fwd)
M_FLIP = np.diag([1.0, -1.0, -1.0])
# OpenGL-style camera axes (look down -Z, y up) -> COLMAP camera axes (look down +Z, y down)
GL_TO_CV = np.diag([1.0, -1.0, -1.0])

SITES = {
    "nthu": "nthu/{scene}/*_{scene}.csv",
    "rutgers": "rutgers/dataset/{scene}/*_{scene}.csv",
}


def quat_to_mat(q):
    """(..., 4) quaternions in x, y, z, w order -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def list_traces(eyenav_root, site, scene):
    """Paths of every participant CSV for one site and scene, sorted by user id."""
    paths = glob.glob(os.path.join(eyenav_root, SITES[site].format(scene=scene)))

    def user_num(p):
        name = os.path.basename(p).split("_")[0]  # "user12"
        return int("".join(c for c in name if c.isdigit()))

    return sorted(paths, key=user_num)


def user_id(path):
    return os.path.basename(path).split("_")[0]


def load_trace(path, drop_invalid_gaze=True):
    """
    Read one participant CSV by column name.

    Returns a dict of numpy arrays, one entry per row (each headset frame has
    two rows, ViewIndex 0 = left eye, 1 = right eye):
      view (N,)  fov (N,4) [left, right, down, up] in radians
      pos (N,3)  quat (N,4) xyzw      eye pose in OpenXR render space
      gaze_pos (N,3)  gaze_quat (N,4)  gaze pose in the same space
      t_ms (N,)  time since recording start in milliseconds
      valid (N,) bool, False when the headset logged no gaze
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty trace: {path}")

    def col(*names):
        return np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float64)

    tcol = "Timestamp" if "Timestamp" in rows[0] else "timestep"
    tr = {
        "view": col("ViewIndex")[:, 0].astype(int),
        "fov": col("FOV1", "FOV2", "FOV3", "FOV4"),
        "pos": col("PositionX", "PositionY", "PositionZ"),
        "quat": col("QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW"),
        "gaze_pos": col("GazePosX", "GazePosY", "GazePosZ"),
        "gaze_quat": col("GazeQX", "GazeQY", "GazeQZ", "GazeQW"),
        "t_ms": col(tcol)[:, 0],
    }
    # The recorder writes pos (0,0,0) and quat (0,0,0,1) when eye tracking is off.
    empty = (np.abs(tr["gaze_pos"]).sum(1) == 0) & (np.abs(tr["gaze_quat"][:, :3]).sum(1) == 0)
    tr["valid"] = ~empty
    if drop_invalid_gaze:
        tr = {k: v[~empty] for k, v in tr.items()}
    return tr


def to_scene(tr):
    """
    Map a loaded trace into the splat's world frame.

    Adds:
      C (N,3)        eye centre in world coords
      R_c2w (N,3,3)  camera-to-world rotation in COLMAP camera convention
                     (x right, y down, z forward), ready for 3DGS cameras
      gaze_o (N,3)   gaze ray origin in world coords
      gaze_d (N,3)   unit gaze direction in world coords
      head_fwd (N,3) unit head-forward direction in world coords
    """
    R_eye = quat_to_mat(tr["quat"])          # OpenXR eye pose, -Z = forward
    R_gaze = quat_to_mat(tr["gaze_quat"])
    out = dict(tr)
    out["C"] = tr["pos"] @ M_FLIP.T
    out["R_c2w"] = M_FLIP @ R_eye @ GL_TO_CV
    out["gaze_o"] = tr["gaze_pos"] @ M_FLIP.T
    fwd = np.array([0.0, 0.0, -1.0])
    out["gaze_d"] = (M_FLIP @ (R_gaze @ fwd)[..., None])[..., 0]
    out["head_fwd"] = (M_FLIP @ (R_eye @ fwd)[..., None])[..., 0]
    return out


def gaze_in_camera(scene_tr):
    """Gaze direction in each row's own camera frame (COLMAP convention, +z forward)."""
    R_w2c = np.transpose(scene_tr["R_c2w"], (0, 2, 1))
    return np.einsum("nij,nj->ni", R_w2c, scene_tr["gaze_d"])


def eccentricity_deg(scene_tr):
    """Angle between gaze and head-forward per row, in degrees."""
    c = np.clip((scene_tr["gaze_d"] * scene_tr["head_fwd"]).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(c))
