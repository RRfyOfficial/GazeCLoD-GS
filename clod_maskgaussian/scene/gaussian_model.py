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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from icecream import ic
from copy import deepcopy

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.mask_activation = nn.LogSoftmax(dim=-1)

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int,enable_opacity_decay_train=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask_score = torch.empty(0)
        self.enable_opacity_decay_train = enable_opacity_decay_train
        if enable_opacity_decay_train:
            self._opacity_dis_decay = torch.empty(0)  # LOD: 距离衰减系数
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask_score,
            self._opacity_dis_decay if hasattr(self, '_opacity_dis_decay') else None,
            self._opacity_dis_decay,  # 添加到capture中
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._mask_score,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        if self.enable_opacity_decay_train:
            self._opacity_dis_decay = model_args.get('_opacity_dis_decay', None)  
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def restore_from_3dgs(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        mask_scores = torch.ones((self._xyz.shape[0], 2), dtype=torch.float, device="cuda") @ torch.tensor([[10.,0.], [0., 1.]], dtype=torch.float, device="cuda")
        self._mask_score = nn.Parameter(mask_scores.requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # Transform the 3dgs state_dict: add pseudo mask info at the 4th position
        state_dict = deepcopy(opt_dict)
        state_dict['state'][6] = state_dict['state'][5]
        state_dict['state'][5] = state_dict['state'][4]
        state_dict['state'][4] = {'step': state_dict['state'][0]['step'],
                                'exp_avg': torch.zeros_like(self._mask_score),
                                'exp_avg_sq': torch.zeros_like(self._mask_score)}
        tmp_param_group = deepcopy(state_dict['param_groups'][3])
        tmp_param_group['lr'] = training_args.mask_lr
        tmp_param_group['name'] = 'mask'
        tmp_param_group['params'] = [4]
        state_dict['param_groups'].insert(4, tmp_param_group)
        state_dict['param_groups'][5]['params'] = [5]
        state_dict['param_groups'][6]['params'] = [6]
        self.optimizer.load_state_dict(state_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_mask(self):
        return nn.functional.gumbel_softmax(self.mask_activation(self._mask_score), hard=True)[:, 0:1]
    
    @property
    def get_opacity_dis_decay(self):
        if not hasattr(self, '_opacity_dis_decay'):
            return None
        return self._opacity_dis_decay
    
    def get_opacity_decay(self, distances, viewpoint_camera, virtual_depth_scale=1.0, enable_view_distance=False):
        """

        """
        scaled_distances = distances * virtual_depth_scale

        
        if self.enable_opacity_decay_train and hasattr(self, '_opacity_dis_decay') and self._opacity_dis_decay is not None:
            # 
            if self._opacity_dis_decay.shape[0] != distances.shape[0]:
                # create a warning log
                print(f"Warning: opacity_dis_decay shape mismatch. Expected {distances.shape[0]}, got {self._opacity_dis_decay.shape[0]}. Using default values.")
                if self._opacity_dis_decay.shape[0] < distances.shape[0]:
                    opacity_dis_decay = torch.ones_like(distances)
                    opacity_dis_decay[:self._opacity_dis_decay.shape[0]] = self._opacity_dis_decay
                else:
                    opacity_dis_decay = self._opacity_dis_decay[:distances.shape[0]]
            else:
                opacity_dis_decay = self._opacity_dis_decay
        else:
            # 
            opacity_dis_decay = torch.ones_like(distances)
        # print("opacity_dis_decay:", opacity_dis_decay)
        variance = 2 * torch.relu(opacity_dis_decay)**2 + 1e-6
        decay_factor = torch.exp(-(scaled_distances**2) / variance)
        if enable_view_distance:
            # 
            z_axis = torch.tensor(viewpoint_camera.R[:,2], dtype=torch.float32).to(self._xyz.device)
            z_axis = torch.nn.functional.normalize(z_axis, dim=0)
            
            assert self._xyz.dtype == torch.float32, "self._xyz must be float32"
            points_to_camera = self._xyz - viewpoint_camera.camera_center.to(torch.float32)
            points_to_camera_norm = torch.nn.functional.normalize(points_to_camera, dim=1)
            
            FoVx = torch.tensor(viewpoint_camera.FoVx, dtype=torch.float32)
            FoVy = torch.tensor(viewpoint_camera.FoVy, dtype=torch.float32)
            fov_radians = max(FoVx, FoVy) / 180.0
            
            cos_theta = torch.sum(points_to_camera_norm * z_axis, dim=1, keepdim=True)
            angle_decay = 1 + (torch.relu(cos_theta) * fov_radians - 0.5) * 0.2
            decay_factor = decay_factor * angle_decay
        decay_factor = torch.nan_to_num(decay_factor, nan=0.0, posinf=1.0, neginf=0.0)
        # print("decay_factor.max:", decay_factor.max().item(), "decay_factor.min:", decay_factor.min().item())

        return decay_factor

    def get_lod_filter_mask(self, viewpoint_camera, enable_distance_culling, enable_view_distance, virtual_depth_scale=1.0, opacity_threshold=0.005):
        """
        compute LOD filtering mask and decayed opacity based on distance to camera
        """
        if not enable_distance_culling or opacity_threshold<=0.0:
            # if not enable_distance_culling or opacity_threshold<=0.0 --- IGNORE ---
            return None, self.get_opacity

        # 1. distance to camera
        distances = torch.norm(self._xyz.detach() - viewpoint_camera.camera_center, dim=1, keepdim=True)
        centroid = torch.mean(self._xyz, dim=0)
        scene_scale = torch.mean(torch.norm(self._xyz - centroid, dim=1))
        distances = distances / (scene_scale +1e-6) # normalize to scene scale
        # 2. compute decay factor
        decay_factor = self.get_opacity_decay(distances, viewpoint_camera,virtual_depth_scale,enable_view_distance=enable_view_distance)
        # print("LOD decay factor shape:", decay_factor.shape)
        # print("LOD decay factor:", decay_factor)
        # print("min decay factor:", decay_factor.min().item(), "max decay factor:", decay_factor.max().item())
        # 3. compute decayed opacity
        decayed_opacity = self.get_opacity * decay_factor

        # 4. generate mask
        if enable_distance_culling:
            lod_mask = decayed_opacity.squeeze(-1) > opacity_threshold*virtual_depth_scale#opacity_threshold+(1-opacity_threshold)*(1-1/virtual_depth_scale)#！
            num_points = decayed_opacity.shape[0]    # 
            k = min(5000, num_points)  
            valid_points = lod_mask.sum().item()  # 
            # 
            # print(f"Filtering: {num_points} points -> {valid_points} points (threshold={opacity_threshold})")
            
            # 
            if valid_points < k and num_points > 0:
                # print(f"Warning: Only {valid_points} points remain after filtering. Selecting top {k} points.")
                # 
                _, top_k_indices = torch.topk(decayed_opacity.squeeze(-1), k)
                new_mask = torch.zeros(num_points, dtype=torch.bool, device=self._xyz.device)
                new_mask[top_k_indices] = True
                lod_mask = new_mask
        else:
            # 
            lod_mask = torch.ones_like(decayed_opacity.squeeze(-1), dtype=torch.bool)

        return lod_mask, decayed_opacity



    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        mask_scores = torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda") @ torch.tensor([[10.,0.], [0., 1.]], dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._mask_score = nn.Parameter(mask_scores.requires_grad_(True))
        if self.enable_opacity_decay_train:
            opacity_dis_decay = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            self._opacity_dis_decay = nn.Parameter(opacity_dis_decay.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._mask_score], 'lr':training_args.mask_lr, "name": "mask"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        
        # Add opacity distance decay parameter if enabled
        if self.enable_opacity_decay_train:
            l.append({'params': [self._opacity_dis_decay], 'lr': training_args.opacity_dis_decay_lr, "name": "opacity_dis_decay"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._mask_score.shape[1]):
            l.append('masks_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # LOD
        if self.enable_opacity_decay_train:
            l.append('opacity_dis_decay')
        return l
    
    def construct_list_of_attributes_default(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # LOD: 
        if self.enable_opacity_decay_train:
            l.append('opacity_dis_decay')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        masks = self._mask_score.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.enable_opacity_decay_train:
            opacity_dis_decay = self._opacity_dis_decay.detach().cpu().numpy()  # LOD

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.enable_opacity_decay_train:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, opacity_dis_decay), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_ply_default(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self.get_mask
        mask = mask.to(torch.bool).squeeze(1)
        # self._xyz = self._xyz[mask]
        # self._features_dc = self._features_dc[mask]
        # self._features_rest = self._features_rest[mask]
        # self._opacity = self._opacity[mask]
        # self._scaling = self._scaling[mask]
        # self._rotation = self._rotation[mask]

        xyz = self._xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[mask].detach().cpu().numpy()
        scale = self._scaling[mask].detach().cpu().numpy()
        rotation = self._rotation[mask].detach().cpu().numpy()
        if self.enable_opacity_decay_train:
            opacity_dis_decay = self._opacity_dis_decay[mask].detach().cpu().numpy()  # LOD

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_default()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.enable_opacity_decay_train:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, opacity_dis_decay), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        ic(f"Saving Gaussians. Number of Gaussians: {self._xyz[mask].shape[0]}")

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        masks = np.zeros((xyz.shape[0], 2))
        masks[:, 0] = np.asarray(plydata.elements[0]["masks_0"])
        masks[:, 1] = np.asarray(plydata.elements[0]["masks_1"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
       
        # LOD: 
        if self.enable_opacity_decay_train:
            opacity_dis_decay = np.ones((xyz.shape[0], 1))  # 
            if "opacity_dis_decay" in [p.name for p in plydata.elements[0].properties]:
                opacity_dis_decay = np.asarray(plydata.elements[0]["opacity_dis_decay"])[..., np.newaxis]
            self._opacity_dis_decay = nn.Parameter(torch.tensor(opacity_dis_decay, dtype=torch.float, device="cuda").requires_grad_(True))  # LOD
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._mask_score = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask_score = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # LOD: 
        if self.enable_opacity_decay_train:
            if "opacity_dis_decay" in optimizable_tensors:
                self._opacity_dis_decay = optimizable_tensors["opacity_dis_decay"]
            else:
                # 
                self._opacity_dis_decay = self._opacity_dis_decay[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] in tensors_dict:  # LOD: 
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_masks, new_scaling, new_rotation, new_opacity_dis_decay=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "mask": new_masks,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        
        # Add opacity distance decay if provided
        if new_opacity_dis_decay is not None:
            d["opacity_dis_decay"] = new_opacity_dis_decay

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask_score = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # Update opacity distance decay if it exists
        if self.enable_opacity_decay_train:
            self._opacity_dis_decay = optimizable_tensors["opacity_dis_decay"]  # LOD

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_mask = self._mask_score[selected_pts_mask].repeat(N,1)
        if self.enable_opacity_decay_train:
            new_opacity_dis_decay = self._opacity_dis_decay[selected_pts_mask].repeat(N,1)  # LOD
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,new_mask, new_scaling, new_rotation, new_opacity_dis_decay)
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,new_mask, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_masks = self._mask_score[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.enable_opacity_decay_train:
            new_opacity_dis_decay = self._opacity_dis_decay[selected_pts_mask] # LOD
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_masks ,  new_scaling, new_rotation, new_opacity_dis_decay)
        else:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,new_masks,  new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if grads.shape[0] != self.get_xyz.shape[0]:
            print(f"Warning: grads形状 {grads.shape[0]} 与xyz形状 {self.get_xyz.shape[0]} 不匹配。调整grads大小。")
            full_grads = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            copy_size = min(grads.shape[0], full_grads.shape[0])
            full_grads[:copy_size] = grads[:copy_size]
            grads = full_grads

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        sample_times = 10
        batched_mask_score = self._mask_score.unsqueeze(0).repeat(sample_times, 1, 1)
        res = nn.functional.gumbel_softmax(self.mask_activation(batched_mask_score), hard=True)[:, :, 0].sum(0)
        prune_mask = torch.logical_or(prune_mask, res==0.)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def mask_prune(self):
        sample_times = 10
        batched_mask_score = self._mask_score.unsqueeze(0).repeat(sample_times, 1, 1)
        res = nn.functional.gumbel_softmax(self.mask_activation(batched_mask_score), hard=True)[:, :, 0].sum(0)
        prune_mask = (res==0.).squeeze()
        # prune_mask = (torch.sigmoid(self._mask_score[:, 1:2]) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
