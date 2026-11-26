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
import os,time,glob
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_orig_3dgs,render_gs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
# from scene.gaussian_model_3dgs import GaussianModel
from scene.gaussian_model_copy import GaussianModel
from utils.image_utils import psnr
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, separate_sh, opt,renders_dir_suffix="renders2"):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), renders_dir_suffix)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    rv = int(opt.rvd_distance_scale)
    decayOpa_str = "decayOpa" if pipeline.enable_decay_opacity_render else "opa"
    decayTrain_str = "_decayTrain" if opt.enable_opacity_decay_train else "_noDecayTrain"
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        idx_str = '{0:05d}'.format(idx)
        pattern = f"{idx_str}*_rvd{rv}*{decayOpa_str}*{decayTrain_str}*.png"
        matching_files = glob.glob(os.path.join(render_path, pattern))
        if len(matching_files) > 0:
            print(f"file {pattern} exists, skip rendering")
            continue  

        start_time = time.time()
        render_pkg = render_gs(view, gaussians, pipeline, background, virtual_scale = opt.rvd_distance_scale)
        end_time = time.time()
        used_time = end_time - start_time
        rendering = render_pkg["render"]
        lod_mask = render_pkg.get("lod_mask", None)
        points_used = render_pkg.get("points_used")

        gt = view.original_image[0:3, :, :]
        psnr_val = psnr(rendering.squeeze(), gt.squeeze()).mean()   
        filename_suffix = f"_rvd{int(opt.rvd_distance_scale)}"
        filename_suffix += f"_psnr{psnr_val:.2f}"
        filename_suffix += f"_{lod_mask.shape[0]}_{lod_mask.sum().item()}_{int(lod_mask.sum().item()/lod_mask.shape[0]*1000)}_" if lod_mask is not None else f"_{points_used}_{points_used}_1000_"
        filename_suffix += f"decayOpa" if pipeline.enable_decay_opacity_render else "opa"
        filename_suffix += f"_decayTrain" if opt.enable_opacity_decay_train else "_noDecayTrain"
        filename_suffix += f"_time{used_time:.2f}s"
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + filename_suffix + ".png"))
        if rv == 1:
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, opt: OptimizationParams,renders_dir_suffix="renders2"):
    print(f"opt para is {opt}")
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree,enable_opacity_decay_train=opt.enable_opacity_decay_train)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, separate_sh, opt, renders_dir_suffix)

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,  opt=op.extract(args), renders_dir_suffix=args.renders_dir_suffix)