"""
Session 5: a per-Gaussian level-of-detail knob driven by gaze heat, plus fair
budget matching.

CLoD-GS has one global "pretend distance" knob s. Gaussian i fades as
    alpha_i' = alpha_i * exp(-(d_i s)^2 / (2 ReLU(sigma_i)^2 + 1e-6))
(d_i = distance to the camera / scene size, sigma_i learned per Gaussian) and
is dropped when alpha_i' <= 0.005 s.

GazeCLoD-GS gives every Gaussian its own knob:
    s_i = max(1, s_base * exp(-k h_i))
h_i in [0, 1] is the crowd heat. k = 0 is plain CLoD-GS. Looked-at Gaussians
act as if the camera were closer, so they keep their detail. The clamp at 1
means nothing is ever drawn with more detail than the full-detail reference.

Fair comparison: for each frame, s_base is found by bisection so the gaze
version draws no more Gaussians than uniform CLoD-GS at scale s. Every
Gaussian's keep/drop decision is monotone in its own s_i (fade gets stronger
and the threshold gets higher), so the count falls steadily as s_base grows.
"""
import math

import numpy as np
import torch

OPACITY_THRESHOLD = 0.005  # CLoD-GS default (PipelineParams.opacity_threshold)
TOPK_FALLBACK = 5000       # CLoD-GS keeps the top 5000 if fewer survive


def user_number(uid):
    return int("".join(c for c in str(uid) if c.isdigit()))


def split_teams(npz_path):
    """Odd-numbered and even-numbered users stored in a heat file."""
    users = [str(u) for u in np.load(npz_path)["users"]]
    odd = [u for u in users if user_number(u) % 2 == 1]
    even = [u for u in users if user_number(u) % 2 == 0]
    return odd, even


def team_heat(npz_path, users):
    """Crowd heat of a group = sum of their per-user heat (each user already averaged over samples)."""
    z = np.load(npz_path)
    sel = np.isin(z["users"], list(users))
    if not sel.any():
        raise ValueError(f"none of {users} are in {npz_path}")
    return z["heat"][sel].sum(0), [str(u) for u in z["users"][sel]]


def heat_to_h(H, gamma=0.5, q=0.99):
    """Raw heat -> h in [0, 1]: divide by the q-quantile of positive heat, clip, then ^gamma (<1 lifts weak heat)."""
    H = torch.as_tensor(H, dtype=torch.float32, device="cuda")
    pos = H[H > 0]
    if pos.numel() == 0:
        return torch.zeros_like(H)
    ref = torch.quantile(pos[:: max(1, pos.numel() // 1_000_000)], q)
    return (H / ref).clamp(0, 1) ** gamma


def shuffled(h, seed=0):
    """Same heat values on random Gaussians (control: non-uniform but not gaze-informed)."""
    g = torch.Generator().manual_seed(seed)
    return h[torch.randperm(h.numel(), generator=g).to(h.device)]


def s_vector(h, s_base, k):
    """Per-Gaussian knob s_i = max(1, s_base * exp(-k h_i)), shape (N,)."""
    return torch.clamp(s_base * torch.exp(-k * h), min=1.0)


class LodState:
    """
    CLoD-GS's keep/drop rule for one camera, with everything that does not
    depend on s precomputed, so trying many s values is cheap. Mirrors
    GaussianModel.get_lod_filter_mask line by line.
    """

    def __init__(self, gs, cam):
        xyz = gs.get_xyz.detach()
        centroid = torch.mean(xyz, dim=0)
        scene_scale = torch.mean(torch.norm(xyz - centroid, dim=1))
        self.d = (torch.norm(xyz - cam.camera_center, dim=1) / (scene_scale + 1e-6))
        self.var = 2 * torch.relu(gs._opacity_dis_decay.detach().reshape(-1)) ** 2 + 1e-6
        self.opacity = gs.get_opacity.detach().reshape(-1)

    def keep(self, s):
        """Boolean (N,) mask of Gaussians CLoD-GS would draw at scale s (float or (N,) tensor)."""
        scaled = self.d * s
        decay = torch.nan_to_num(torch.exp(-(scaled ** 2) / self.var), nan=0.0, posinf=1.0, neginf=0.0)
        faded = self.opacity * decay
        mask = faded > OPACITY_THRESHOLD * s
        if int(mask.sum()) < TOPK_FALLBACK:
            mask = torch.zeros_like(mask)
            mask[torch.topk(faded, min(TOPK_FALLBACK, faded.numel())).indices] = True
        return mask

    def count(self, s):
        return int(self.keep(s).sum())


def match_budget(state, h, k, target, s_start, iters=50):
    """
    Smallest s_base whose gaze knob draws <= target Gaussians.
    At s_base = s_start (the uniform s) every s_i <= s_start, so the gaze
    version keeps at least as many; we bisect upward on log(s_base).
    Returns (s_base, count).
    """
    lo, hi = s_start, s_start * 2.0
    while state.count(s_vector(h, hi, k)) > target:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            break
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if state.count(s_vector(h, mid, k)) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-5:
            break
    return hi, state.count(s_vector(h, hi, k))
