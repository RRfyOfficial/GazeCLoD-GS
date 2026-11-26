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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           separate_sh = False, override_color = None, use_trained_exp=False,virtual_scale=1.0,start_cull=True,topk_big=None,dis_opa_topk=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    lod_level: -1 for no LOD, 0+ for specific LOD levels
    virtual_scale: Scale factor for distances (for virtual distance rendering)
    fresnel_decay: Optional tensor with Fresnel-based opacity decay factors
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # set initial opacities
    opacity = pc.get_opacity

    # LOD: apply distance-based opacity decay if enabled
    points_used = pc.get_xyz.shape[0]  # default to total points
    # compute LOD mask and decayed opacity
    lod_mask, decayed_opacity = pc.get_lod_filter_mask(
        viewpoint_camera, 
        pipe.enable_distance_culling and start_cull,
        pipe.enable_view_distance,
        virtual_depth_scale=virtual_scale,
        opacity_threshold=pipe.opacity_threshold,
        topk_big=topk_big,
        dis_opa_topk=dis_opa_topk
    )
    # print("shape of opacity before decay:", opacity.shape)
    # print("shape of lod_mask:", lod_mask.shape," number of True in lod_mask:", lod_mask.sum().item())
    # print("ratio of visible points to total points:", lod_mask.sum().item()/lod_mask.shape[0])
    # apply decayed opacity
    if pipe.enable_decay_opacity_render:
        opacity = decayed_opacity
    
    # filter points based on LOD mask
    if pipe.enable_distance_culling and start_cull and lod_mask is not None:
        means3D = means3D[lod_mask]
        means2D = means2D[lod_mask]
        opacity = opacity[lod_mask]
        points_used = lod_mask.sum().item() 
    # print("shape of opacity after decay and filtering:", opacity.shape)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        if lod_mask is not None and pipe.enable_distance_culling and start_cull:
            cov3D_precomp = pc.get_covariance(scaling_modifier)[lod_mask]
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if lod_mask is not None and pipe.enable_distance_culling and start_cull:
            scales = pc.get_scaling[lod_mask]
            rotations = pc.get_rotation[lod_mask]
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            features = pc.get_features if lod_mask is None or not pipe.enable_distance_culling or not start_cull else pc.get_features[lod_mask]
            xyz_for_sh = pc.get_xyz if lod_mask is None or not pipe.enable_distance_culling or not start_cull else pc.get_xyz[lod_mask]
            
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (xyz_for_sh - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                if lod_mask is not None and pipe.enable_distance_culling and start_cull:
                    dc, shs = pc.get_features_dc[lod_mask], pc.get_features_rest[lod_mask]
                else:
                    dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                if lod_mask is not None and pipe.enable_distance_culling and start_cull:
                    shs = pc.get_features[lod_mask]
                else:
                    shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    
    # LOD: adjust visibility filter based on LOD mask
    visibility_filter = (radii > 0).nonzero()
    visibility_filter_M = visibility_filter.clone()  # For M points after culling
    
    if lod_mask is not None and pipe.enable_distance_culling and start_cull:
        # map visibility filter back to original indices
        original_indices = torch.nonzero(lod_mask).squeeze()
        if visibility_filter.numel() > 0:
            visibility_filter = original_indices[visibility_filter.squeeze()]
            visibility_filter = visibility_filter.unsqueeze(1) if visibility_filter.dim() == 1 else visibility_filter
    
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "visibility_filter_M": visibility_filter_M,  # For M points
        "radii": radii,
        "depth" : depth_image,
        "lod_mask": lod_mask,  # add LOD mask as output
        "points_used": points_used  # record number of points used after LOD filtering
        }
    
    return out
