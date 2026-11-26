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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Using fused ssim")
except:
    FUSED_SSIM_AVAILABLE = False
    print("dont Using ssim")

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def compute_global_optimal_distance_loss(gaussians, viewpoint_camera):
    """
    Computes a regularization loss that encourages a diverse distribution of distance decay factors
    not used in the paper.
    """
    distances = torch.norm(gaussians.get_xyz - viewpoint_camera.camera_center.unsqueeze(0), dim=1)
    
    opacity_dis_decay = gaussians.get_opacity_dis_decay.squeeze()
    distance_variance_loss = -torch.var(opacity_dis_decay)  # maximize variance of distance decay factors
    
    return distance_variance_loss

def downsample_image(gt_image, x):
    import torch.nn.functional as F
    # compute new size
    virtual_scale = 1/(1+(x-1)**0.3) #scale=5,0.4;7>>0.37;3>>0.45;2>>0.5
    new_size = (int(gt_image.shape[1] * virtual_scale), int(gt_image.shape[2] * virtual_scale))
    downsampled_image = F.interpolate(gt_image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False)
    downsampled_image = F.interpolate(downsampled_image, size=(gt_image.shape[1], gt_image.shape[2]), mode='bilinear', align_corners=False)
    return downsampled_image.squeeze(0)


def compute_point_count_regularization(points_used, total_points, virtual_scale):
    """
    Args:
        points_used: the amount of points used in the current iteration
        total_points: the total amount of points in the model    
    Returns:
    loss for point count regularization
    """
    # ratio of used points to total points
    points_ratio = points_used / total_points
    

    target_ratio = 1.0 / virtual_scale**1.5  # target ratio decreases with increasing virtual scale
    
    # loss for point count regularization
    loss = torch.relu(torch.tensor(points_ratio) - torch.tensor(target_ratio))**2
    
    # scale coefficient: 0 when scale=1, increases with increasing scale
    scale_weight = 1.0 *(virtual_scale - 1.0)**2
    
    return loss * scale_weight

import csv
from datetime import datetime

def savepara(args, opt, pipe):
    # save parameters
    parameters_to_save = {
        "model_name": args.model_path.split("/")[-3:],

        "enable_opacity_decay_train": opt.enable_opacity_decay_train,
        "opacity_dis_decay_lr": opt.opacity_dis_decay_lr,
        "opacity_dis_decay_init": opt.opacity_dis_decay_init,
        "rvd_probability": opt.rvd_probability,
        "rvd_distance_scale": opt.rvd_distance_scale,
        "global_optimal_distance_weight": opt.global_optimal_distance_weight,

        "enable_decay_opacity_render": pipe.enable_decay_opacity_render,
        "enable_distance_culling": pipe.enable_distance_culling,
        "opacity_threshold": pipe.opacity_threshold,
        "enable_view_distance": pipe.enable_view_distance,

        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": os.popen('git rev-parse HEAD').read().strip()  

    }

    save_file = "training_parameters2.csv"
    
    file_exists = os.path.isfile(save_file)
    with open(save_file, 'a', newline='') as csvfile:
        fieldnames = list(parameters_to_save.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  
        writer.writerow(parameters_to_save)  

    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    savepara(dataset,opt,pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type,opt.enable_opacity_decay_train)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Multi-Scale Training
        losses = []
        
        multi_scale_enabled = opt.rvd_distance_scale > 1.0 and iteration > opt.iter_start_culling
        if opt.early_stop_culling:
            multi_scale_enabled = multi_scale_enabled and iteration < opt.iter_start_early_stop_culling

        batch_size = opt.multi_scale_batch if hasattr(opt, 'multi_scale_batch') else 1
        
        if multi_scale_enabled:
            # Generate multiple scales from coarse to fine (highest to lowest virtual scale)
            scales = []
            # Create a batch of evenly distributed scales from max scale to 1.0
            if batch_size > 1:
                for i in range(batch_size):
                    if i == batch_size - 1:  # Last item is always 1.0 (full resolution)
                        scales.append(1.0)
                    else:
                        # Distribute scales from max to near 1.0
                        scale_value = opt.rvd_distance_scale - ((i+(torch.rand(1).item()-0.5)) * (opt.rvd_distance_scale - 1.0) / (batch_size - 1))
                        scales.append(scale_value)
            else:
                # Original random behavior if batch_size is 1
                use_rvd = torch.rand(1).item() < opt.rvd_probability
                if use_rvd:
                    scales.append(1.0 + torch.rand(1).item() * (opt.rvd_distance_scale-1))
                else:
                    scales.append(1.0)
        else:
            # If multi-scale not enabled, just use scale 1.0
            scales = [1.0]
        #计算weight
        scale_weights = []
        for virtual_scale in scales:
            if opt.weight_scale:
                if opt.weight_type ==1:
                # Progressive weighting: higher resolution (smaller scale) gets higher weight
                    scale_weight = 1.0 - virtual_scale/opt.rvd_distance_scale * 0.5
                elif opt.weight_type == 2:
                    # Exponential weighting: higher resolution (smaller scale) gets higher weight
                    scale_weight = 1.0 + virtual_scale/opt.rvd_distance_scale
                else:
                    scale_weight = 1.0
                scale_weights.append(scale_weight)
            else:
                scale_weights.append(1.0)
        scale_weights_sum = sum(scale_weights)
        # Loop through all scales (coarse to fine) for progressive training
        accumulated_loss = 0.0
        for i, virtual_scale in enumerate(scales):

            start_cull=iteration > opt.iter_start_culling
            if opt.early_stop_culling:
                start_cull = start_cull and iteration < opt.iter_start_early_stop_culling

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                             use_trained_exp=dataset.train_test_exp, 
                             separate_sh=SPARSE_ADAM_AVAILABLE,
                             virtual_scale=virtual_scale,
                             start_cull=start_cull)
            
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            visibility_filter_M = render_pkg["visibility_filter_M"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Calculate loss for this scale
            gt_image = viewpoint_cam.original_image.cuda()
            
            # Optional downsampling for lower resolution (generally not recommended)
            # if virtual_scale > 1.0 and opt.enable_downsampling:
            #     gt_image = downsample_image(gt_image, virtual_scale)
                
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Add point count regularization for all scales except the full resolution one
            if virtual_scale > 1.0 and opt.l_reg > 0.01 and iteration > opt.iter_start_culling:
                # print("Scale {:.2f}: Using {}/{} points".format(virtual_scale, render_pkg["points_used"], gaussians.get_xyz.shape[0]))
                points_used = render_pkg["points_used"]
                total_points = gaussians.get_xyz.shape[0]
                point_count_reg = compute_point_count_regularization(points_used, total_points, virtual_scale)
                loss += opt.l_reg*point_count_reg
                
                if tb_writer and iteration % 100 == 0 and i == 0:  # Log only for the first scale
                    tb_writer.add_scalar(f'train/points_ratio_scale_{virtual_scale:.1f}', points_used / total_points, iteration)
                    tb_writer.add_scalar(f'train/point_count_reg_loss_scale_{virtual_scale:.1f}', point_count_reg.item(), iteration)

            # Add global distance loss if enabled
            # if opt.enable_opacity_decay_train and opt.global_optimal_distance_weight > 0:
            #     global_distance_loss = compute_global_optimal_distance_loss(gaussians, viewpoint_cam)
            #     loss += opt.global_optimal_distance_weight * global_distance_loss

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            # Optional: weight different scales
            if opt.loss_mean:
                loss = loss * scale_weights[i]/scale_weights_sum
            else:
                loss = loss * scale_weights[i]

            # Store loss for logging
            losses.append(loss.item())
            
            # Backward pass based on batch mode
            if not hasattr(opt, 'batch_mode') or opt.batch_mode == "per_scale":
                # Original mode: backward pass for each scale separately
                loss.backward()
            else:
                # Accumulate loss for all scales and do a single backward pass
                if i < len(scales) - 1:
                    # Accumulate loss without backward
                    accumulated_loss += loss
                else:
                    # On the last scale, do the backward pass with accumulated loss
                    if accumulated_loss > 0:
                        # Add the final loss to accumulated loss
                        accumulated_loss += loss
                        accumulated_loss.backward()
                    else:
                        # If there's only one scale or accumulated_loss is still 0
                        loss.backward()
            
            # Update densification stats for the last (highest resolution) scale
            if i == len(scales) - 1 or virtual_scale == 1.0:
                # Keep track of max radii in image-space for pruning
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter_M])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # For logging/display only
            if i == 0:
                first_loss = loss.item()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter_M])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    print("######Using Sparse Adam optimizer#######")
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000,15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
