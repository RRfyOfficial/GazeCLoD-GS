#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
import time
import glob
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from PIL import Image                                   # add


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, opt,
               renders_dir_suffix="renders2", topk_big=None, dis_opa_topk=False):

    render_path = os.path.join(model_path, name, f"ours_{iteration}", renders_dir_suffix)
    gts_path    = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path,   exist_ok=True)

    rv              = int(opt.rvd_distance_scale)
    decayOpa_str    = "decayOpa" if pipeline.enable_decay_opacity_render else "opa"
    decayTrain_str  = "_decayTrain" if opt.enable_opacity_decay_train else "_noDecayTrain"
    if topk_big is not None:
        point_ratio = f"_topk{topk_big:.2f}"
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        idx_str  = f'{idx:05d}'
        pattern  = f"{idx_str}*_rvd{rv}*{decayOpa_str}*{decayTrain_str}*.png"
        if topk_big is not None:
            pattern = f"{idx_str}*_rvd{rv}*{point_ratio}{decayOpa_str}*{decayTrain_str}*.png"
        paths    = glob.glob(os.path.join(render_path, pattern))

        skip_render = True            
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.verify()       
            except Exception as e:     
                print(f"[Corrupted] {p} ({e}).  Will re-render.")
                try: os.remove(p)
                except OSError: pass
                skip_render = False
                break

        if paths and skip_render:
            # all paths exist and are valid -> skip
            print(f"{pattern} exists and is valid, skip.")
            continue
        # --------------------------------------

        #========== rendering ==========
        torch.cuda.synchronize(); t0 = time.time()
        render_pkg  = render(view, gaussians, pipeline, background,
                             use_trained_exp=train_test_exp,
                             separate_sh=separate_sh,
                             virtual_scale=opt.rvd_distance_scale,
                             topk_big=topk_big,
                             dis_opa_topk=dis_opa_topk)
        torch.cuda.synchronize(); t1 = time.time()
        used_time   = t1 - t0

        rendering   = render_pkg["render"]
        lod_mask    = render_pkg.get("lod_mask", None)
        points_used = render_pkg.get("points_used")
        gt          = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt        = gt[..., gt.shape[-1] // 2:]

        psnr_val = psnr(rendering.squeeze(), gt.squeeze()).mean()

        # name file
        filename_suffix  = f"_rvd{rv}"
        filename_suffix += f"_psnr{psnr_val:.2f}"
        if lod_mask is not None:
            filename_suffix += f"_{lod_mask.shape[0]}_{lod_mask.sum().item()}_" \
                               f"{int(lod_mask.sum().item()/lod_mask.shape[0]*1000)}_"
        else:
            filename_suffix += f"_{points_used}_{points_used}_1000_"
        if topk_big is not None:
            filename_suffix += f"_topk{topk_big:.2f}"
        filename_suffix += decayOpa_str
        filename_suffix += decayTrain_str
        filename_suffix += f"_time{used_time:.2f}s"

        save_name = os.path.join(render_path, f"{idx_str}{filename_suffix}.png")
        torchvision.utils.save_image(rendering, save_name)

        if rv == 1:
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx_str}.png"))

        # if i==5:
        #     break

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, opt: OptimizationParams,
    renders_dir_suffix="renders2", topk_big=None, dis_opa_topk=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,enable_opacity_decay_train =opt.enable_opacity_decay_train)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
        # if not skip_train:
        #     render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, opt)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, 
            separate_sh, opt, renders_dir_suffix, topk_big=topk_big, dis_opa_topk=dis_opa_topk)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--renders_dir_suffix", default="renders2", type=str)
    parser.add_argument("--topk_big", default=None, type=float)
    parser.add_argument("--dis_opa_topk", action="store_true")
    args = get_combined_args(parser)
    args2 = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,  
    opt=op.extract(args), renders_dir_suffix=args.renders_dir_suffix, topk_big=args2.topk_big, dis_opa_topk=args2.dis_opa_topk)