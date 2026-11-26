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
from utils.general_utils import safe_state
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

simple_scheduler = lambda value, iter, start, end: value if start <= iter <= end else 0


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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,opt.enable_opacity_decay_train)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Multi-Scale Training
        losses = []
        
        multi_scale_enabled = hasattr(opt, 'rvd_distance_scale') and opt.rvd_distance_scale > 1.0 and iteration > opt.iter_start_culling
        if hasattr(opt, 'early_stop_culling') and opt.early_stop_culling:
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
        total_points_used = 0
        last_mask = None
        
        for i, virtual_scale in enumerate(scales):

            start_cull = iteration > opt.iter_start_culling
            if hasattr(opt, 'early_stop_culling') and opt.early_stop_culling:
                start_cull = start_cull and iteration < opt.iter_start_early_stop_culling

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, virtual_scale=virtual_scale, start_cull=start_cull)
            image, viewspace_point_tensor, visibility_filter, radii, mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["mask"]
            visibility_filter_M = render_pkg["visibility_filter_M"]
            # Get LOD information
            lod_mask = render_pkg.get("lod_mask", None)
            points_used = render_pkg.get("points_used", gaussians.get_xyz.shape[0])
            
            # Downsample GT image if enabled
            current_gt_image = viewpoint_cam.original_image.cuda()
            # if hasattr(opt, 'enable_downsampling') and opt.enable_downsampling and virtual_scale > 1.0:
            #     # Downsample GT image proportionally to virtual scale
            #     scale_factor = 1.0 / virtual_scale
            #     H, W = current_gt_image.shape[1], current_gt_image.shape[2]
            #     new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            #     current_gt_image = torch.nn.functional.interpolate(
            #         current_gt_image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
            #     ).squeeze(0)
            #     # Also need to resize rendered image
            #     image = torch.nn.functional.interpolate(
            #         image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
            #     ).squeeze(0)

            # Loss calculation
            Ll1 = l1_loss(image, current_gt_image)
            mask_loss = (torch.mean(mask))**2
            lambda_mask = simple_scheduler(opt.lambda_mask, iteration, opt.mask_from_iter, opt.mask_until_iter)
            scale_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, current_gt_image)) + lambda_mask * mask_loss
            
            # Regularization to encourage fewer points at higher virtual scales
            if virtual_scale > 1.0 and opt.l_reg>0.01 and iteration > opt.iter_start_culling:
                # print("Scale {:.2f}: Using {}/{} points".format(virtual_scale, render_pkg["points_used"], gaussians.get_xyz.shape[0]))
                points_used = render_pkg["points_used"]
                total_points = gaussians.get_xyz.shape[0]
                point_count_reg = compute_point_count_regularization(points_used, total_points, virtual_scale)
                scale_loss += opt.l_reg*point_count_reg

            if opt.loss_mean:
                loss = scale_loss * scale_weights[i]/scale_weights_sum
            else:
                loss = scale_loss * scale_weights[i]

            # Accumulate loss
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
            total_points_used += points_used
            last_mask = mask
            # Store for logging
            losses.append(loss.item())

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            avg_points_used = total_points_used // len(scales)
            num_used_gs = int(last_mask.detach().sum()) if last_mask is not None else avg_points_used
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_used_gs": num_used_gs, "avg_pts": avg_points_used})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), num_used_gs)
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
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif iteration % opt.mask_prune_iter == 0:
                gaussians.mask_prune()

            # Optimizer step
            if iteration < opt.iterations:
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, num_used_gs=-1):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('num_used_gs', num_used_gs, iteration)
        tb_writer.add_scalar('GPU_mem', torch.cuda.memory_reserved() / 1024**3, iteration)

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
                    if tb_writer and (idx < 5):
                        if iteration == testing_iterations[-1]:
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 16_000,17_000, 18_000, 19_000, 20_000, 21_000, 22_000, 23_000, 24_000, 25_000, 26_000, 27_000, 28_000, 29_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
