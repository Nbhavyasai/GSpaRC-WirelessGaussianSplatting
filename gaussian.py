####Import the point cloud from initialization.py and define gaussian functions, create the tensors to optimizer#####

import torch
import numpy as np
import torch.nn as nn
import os
from initialization import *
from utils import *
from optimizer import *



def get_signal(gaussian_model, x):
    """
    Compute signal_real and signal_imag for all Gaussians independently.
    Input:
    gaussian_model : GaussianModel class instance
    x : Input tensor to the MLP (x_tx, y_tx, z_tx, azimuth, elevation) (5, 1)
    Output:
    signal_real, signal_imag : Outputs of the emission MLP for each gaussian
    """
    signal_real = torch.zeros((gaussian_model.get_num_gaussians, 1), device="cuda")
    signal_imag = torch.zeros((gaussian_model.get_num_gaussians, 1), device="cuda")
    
    for i in range(gaussian_model.get_num_gaussians):
        signals = gaussian_model.get_emission_mlp[i](x)  # Forward pass for Gaussian i
        signal_real[i], signal_imag[i] = signals[..., 0], signals[..., 1]  # Split outputs

    return signal_real, signal_imag

class GaussianModel:

    def setup_functions(self):
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

    def __init__(self, logger=None, debug=False):
        self._centre = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._emission_mlps = None  # Changed from nn.ModuleList()
        self.spatial_lr_scale = 1.0
        self.mlp_size = [3, 32, 2]
        self.centre_scheduler_args = None
        self.mlp_scheduler_args = None
        self.optimizer = None
        self.centre_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.percent_dense = 0.01
        self.debug = debug
        if self.debug and logger:
            self.densification_logger = logger
            self.densification_logger.write("Densification Log:\n")
        elif self.debug:
            raise ValueError("Logger must be provided when debug mode is enabled.")
        self.setup_functions()
        # Confidence MLP: initialized later in initialize_gaussians()
        self._confidence_mlp = None

    def __repr__(self):
        """
        Return a string representation of the GaussianModel object, including the size,
        value, and gradient of all the parameter tensors.
        """
        def tensor_info(name, tensor):
            return f"{name} (size: {tensor.size()}):\n{tensor}\nGradients:\n{tensor.detach().clone().grad}\n"

        return (tensor_info("Centers", self.get_centre) +
                tensor_info("Scales", self.get_scaling) +
                tensor_info("Rotations", self.get_rotation) +
                tensor_info("Opacities", self.get_opacity))

    @property
    def get_num_gaussians(self):
        return self._centre.shape[0]

    @property
    def get_centre(self):
        return self._centre

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_emission_mlps(self):
        return self._emission_mlps.get_all_params()
    
    @property
    def confidence_mlp(self):
        return self._confidence_mlp
    
    def get_confidence(self, rx_pos):
        """
        Compute confidence score for a given receiver position.
        
        Args:
            rx_pos: (3,) or (B, 3) tensor – receiver position(s)
        Returns:
            confidence: scalar or (B,) tensor, always > 1 (DUSt3R convention)
        """
        if self._confidence_mlp is None:
            # Fallback: return 1.0 (no confidence weighting)
            return torch.ones(1, device=rx_pos.device)
        return self._confidence_mlp(rx_pos)
    
    
    
    def initialize_gaussians(self, points): 
        """Initialize Gaussians using properties from PLY file"""
        centres = torch.tensor(points, dtype=torch.float32, device="cuda")
        num_gaussians = centres.shape[0]
        # centres: (N, 3) tensor on CUDA
        def nearest_neighbor_sqdist(points, batch_size=1024):
            N = points.size(0)
            dist2_min = torch.full((N,), float('inf'), device=points.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                chunk = points[start:end]  # (B, 3)
                # squared distances to all points
                diff = chunk.unsqueeze(1) - points.unsqueeze(0)  # (B, N, 3)
                d2 = (diff ** 2).sum(dim=2)                       # (B, N)
                # mask self-distances
                for i in range(start, end):
                    d2[i - start, i] = float('inf')
                dist2_min[start:end] = d2.min(dim=1).values
            return dist2_min
        dist2 = nearest_neighbor_sqdist(centres, batch_size=1024)
        dist2 = torch.clamp_min(dist2, 1e-7)
        scales =  torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)


        rots = torch.zeros((num_gaussians, 4), device="cuda")
        rots[:, 0] = 1  # identity matrix in the quaternion form
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((num_gaussians, 1), dtype=torch.float, device="cuda"))
        

        # Convert to parameters
        self._centre = torch.nn.Parameter(centres, requires_grad=True)
        self._scaling = torch.nn.Parameter(scales, requires_grad=True)
        self._rotation = torch.nn.Parameter(rots, requires_grad=True)
        self._opacity = torch.nn.Parameter(opacities, requires_grad=True)
        self.max_radii2D = torch.zeros((self.get_centre.shape[0]), device="cuda")
        
        # Initialize emission MLPs
        self._emission_mlps = EmissionMLPs(num_gaussians, 
                                        self.mlp_size[0], 
                                        self.mlp_size[1], 
                                        self.mlp_size[2])
        
        # Initialize confidence MLP (rx_pos → confidence scalar)
        # Hidden size will be set properly in training_setup; use default for now
        self._confidence_mlp = ConfidenceMLP(hidden_size=32)
        
        print(f"Initialized {num_gaussians} Gaussians")

        
    def training_setup(self, optim_params, spatial_lr_scale=None):
        # Set the learning rate for each learnable parameter
        self.centre_gradient_accum = torch.zeros((self.get_centre.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_centre.shape[0], 1), device="cuda")
        l = [
            {'params': [self._centre], 'lr': optim_params.position_lr_init*self.spatial_lr_scale, "name": "centre"},
            {'params': [self._scaling], 'lr': optim_params.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': optim_params.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': optim_params.opacity_lr, "name": "opacity"},
            {'params': [self._emission_mlps.fc1_weights], 'lr': optim_params.emission_mlp_lr, "name": "emission_fc1_weights"},
            {'params': [self._emission_mlps.fc1_bias], 'lr': optim_params.emission_mlp_lr, "name": "emission_fc1_bias"},
            {'params': [self._emission_mlps.fc2_weights], 'lr': optim_params.emission_mlp_lr, "name": "emission_fc2_weights"},
            {'params': [self._emission_mlps.fc2_bias], 'lr': optim_params.emission_mlp_lr, "name": "emission_fc2_bias"}
        ]

        # Add confidence MLP parameters if it exists
        if self._confidence_mlp is not None:
            confidence_lr = getattr(optim_params, 'confidence_lr', 0.001)
            # Re-initialise with the correct hidden size from config
            hidden_size = getattr(optim_params, 'confidence_hidden_size', 32)
            if self._confidence_mlp.fc1.out_features != hidden_size:
                self._confidence_mlp = ConfidenceMLP(hidden_size=hidden_size)
            for name, param in self._confidence_mlp.named_parameters():
                l.append({'params': [param], 'lr': confidence_lr, 
                         "name": f"confidence_{name}"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Add gradient clipping to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['clip_grad_norm'] = 1.0  # Clip gradients to max norm of 1.0

        self.centre_scheduler_args = get_expon_lr_func(lr_init=optim_params.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=optim_params.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=optim_params.position_lr_delay_mult,
                                                    max_steps=optim_params.position_lr_max_steps)
        
        # Add scheduler for MLP parameters
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=optim_params.emission_mlp_lr,
                                                 lr_final=optim_params.emission_mlp_lr / 20.0, # Example: reduce LR by 20x
                                                 lr_delay_mult=optim_params.position_lr_delay_mult, # Use same delay as position
                                                 max_steps=optim_params.position_lr_max_steps) # Use same max steps as position

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        centre_lr = self.centre_scheduler_args(iteration)
        mlp_lr = self.mlp_scheduler_args(iteration) # Calculate MLP LR
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "centre":
                param_group['lr'] = centre_lr
        return centre_lr # Return centre LR for potential logging or other uses

    def reset_opacity(self):
        # Ensure numerical stability when resetting opacity
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        opacities_new = torch.nan_to_num(opacities_new, nan=0.0, posinf=1.0, neginf=0.0)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]        

    # Reinitialize the state when replacing a parameter tensor with a new one.
    def replace_tensor_to_optimizer(self, tensor, name):
        """
        Replaces a tensor in the optimizer with a new tensor and updates the optimizer state.

        Args:
            tensor (torch.Tensor): The new tensor to be added to the optimizer.
            name (str): The name of the parameter group to be replaced.

        Returns:
            dict: A dictionary containing the updated parameter group with the new tensor.
        """        
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None) 
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor, requires_grad=True)
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
                                
    # Names of optimizer param groups that hold per-Gaussian tensors.
    # Any group not in this set (e.g. the confidence MLP) is global and must
    # be skipped during densification / pruning.
    PER_GAUSSIAN_GROUPS = {
        "centre", "scaling", "rotation", "opacity",
        "emission_fc1_weights", "emission_fc1_bias",
        "emission_fc2_weights", "emission_fc2_bias",
    }

    def _prune_optimizer(self, mask):
        """Prunes the optimizer's parameters based on the provided mask.
        Args:
            mask (torch.Tensor): A boolean tensor indicating which elements to keep.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            param_name = group["name"]
            # Skip global (non per-Gaussian) param groups such as the confidence MLP.
            if param_name not in self.PER_GAUSSIAN_GROUPS:
                continue
            param = group["params"][0]
            stored_state = self.optimizer.state.get(param, None)

            # Adjust mask size if it does not match the parameter's first dimension
            if mask.shape[0] != param.shape[0]:
                raise ValueError(f"Mask shape {mask.shape} does not match the first dimension of parameter {param_name} with shape {param.shape}")

            # Apply the mask based on the parameter's dimensions
            if param.dim() == 3:  # For 3D tensors like fc1_weights, fc2_weights
                pruned_param = param[mask, :, :]
            elif param.dim() == 2:  # For 2D tensors like fc1_bias, fc2_bias
                pruned_param = param[mask, :]
            elif param.dim() == 1:  # For 1D tensors like _centre, _scaling, etc.
                pruned_param = param[mask]
            else:
                raise ValueError(f"Unsupported parameter dimension: {param.dim()} for parameter {param_name}")

            # Update the optimizer state and parameter
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[param]
                group["params"][0] = nn.Parameter(pruned_param.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[param_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(pruned_param.requires_grad_(True))
                optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        """Prunes points based on the given mask and updates the corresponding tensors.
        Args:
            mask (torch.Tensor): A boolean tensor indicating which points to prune. 
                                 Points corresponding to `True` values in the mask will be pruned.
        Updates:
            self._centre (torch.Tensor): Pruned tensor of xyz coordinates.
            self._scaling (torch.Tensor): Pruned tensor of scaling factors.
            self._rotation (torch.Tensor): Pruned tensor of rotation values. 
            self._opacity (torch.Tensor): Pruned tensor of opacity values.
            self.centre_gradient_accum (torch.Tensor): Pruned tensor of accumulated gradients for centre.
            self.denom (torch.Tensor): Pruned tensor of denominators.
        """
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._centre = optimizable_tensors["centre"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._emission_mlps.fc1_weights = optimizable_tensors["emission_fc1_weights"]
        self._emission_mlps.fc1_bias = optimizable_tensors["emission_fc1_bias"]
        self._emission_mlps.fc2_weights = optimizable_tensors["emission_fc2_weights"]
        self._emission_mlps.fc2_bias = optimizable_tensors["emission_fc2_bias"]

        self.centre_gradient_accum = self.centre_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Concatenates tensors from a given dictionary to the parameters of the optimizer,
        with strict shape validation to prevent tensor corruption.

        Args:
            tensors_dict (dict): A dictionary where keys are parameter group names and values are 
                                tensors to be concatenated to the optimizer's parameters.

        Returns:
            dict: A dictionary of updated optimizer parameters by name.
        """
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            param_name = group["name"]
            # Skip global (non per-Gaussian) param groups such as the confidence MLP.
            if param_name not in self.PER_GAUSSIAN_GROUPS:
                continue
            assert len(group["params"]) == 1, f"Optimizer group {param_name} must contain a single parameter"

            base_tensor = group["params"][0]
            extension_tensor = tensors_dict[param_name]

            # Sanity check for shape consistency (ignore first dim: batch)
            base_shape = base_tensor.shape[1:]
            ext_shape = extension_tensor.shape[1:]
            if base_shape != ext_shape:
                raise ValueError(f"[cat_tensors_to_optimizer] Shape mismatch in '{param_name}': "
                                f"existing tensor shape {base_tensor.shape}, "
                                f"extension shape {extension_tensor.shape}")

            # Extend optimizer state
            stored_state = self.optimizer.state.get(base_tensor, None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[base_tensor]

            # Concatenate the tensors
            concatenated_tensor = torch.cat((base_tensor, extension_tensor), dim=0).requires_grad_(True)
            group["params"][0] = nn.Parameter(concatenated_tensor)

            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    
    def densification_postfix(self, new_centre, new_scaling, new_rotation, new_opacity, new_emission_fc1_weights, new_emission_fc1_bias, new_emission_fc2_weights, new_emission_fc2_bias, new_tmp_radii):
        """
        Updates the model's internal state with new data and prepares tensors for optimization.

        Args:
            new_xyz (torch.Tensor): New coordinates to be added.
            new_features_dc (torch.Tensor): New direct current features to be added.
            new_features_rest (torch.Tensor): New remaining features to be added.
            new_opacities (torch.Tensor): New opacities to be added.
            new_scaling (torch.Tensor): New scaling factors to be added.
            new_rotation (torch.Tensor): New rotation matrices to be added.
            new_tmp_radii (torch.Tensor): New temporary radii to be added.

        Updates:
            self._xyz (torch.Tensor): Updated coordinates tensor.
            self._features_dc (torch.Tensor): Updated direct current features tensor.
            self._features_rest (torch.Tensor): Updated remaining features tensor.
            self._opacity (torch.Tensor): Updated opacities tensor.
            self._scaling (torch.Tensor): Updated scaling factors tensor.
            self._rotation (torch.Tensor): Updated rotation matrices tensor.
            self.tmp_radii (torch.Tensor): Concatenated temporary radii tensor.
            self.xyz_gradient_accum (torch.Tensor): Zero-initialized gradient accumulation tensor.
            self.denom (torch.Tensor): Zero-initialized denominator tensor.
            self.max_radii2D (torch.Tensor): Zero-initialized maximum radii tensor.
        """
        d = {
            "centre": new_centre,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "emission_fc1_weights": new_emission_fc1_weights ,  # Add emission MLP keys
            "emission_fc1_bias": new_emission_fc1_bias, 
            "emission_fc2_weights": new_emission_fc2_weights,
            "emission_fc2_bias": new_emission_fc2_bias,
        }  
       
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._centre = optimizable_tensors["centre"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._emission_mlps.fc1_weights = optimizable_tensors["emission_fc1_weights"]
        self._emission_mlps.fc1_bias = optimizable_tensors["emission_fc1_bias"]
        self._emission_mlps.fc2_weights = optimizable_tensors["emission_fc2_weights"]
        self._emission_mlps.fc2_bias = optimizable_tensors["emission_fc2_bias"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.centre_gradient_accum = torch.zeros((self.get_centre.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_centre.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_centre.shape[0]), device="cuda")
        
       

    
    # Define densify_split function
    def densify_and_split(self, grads, grad_threshold, scaling_threshold, N=2):
        """
        Densifies and splits the points in the scene based on the gradient condition.

        Args:
            grads (torch.Tensor): Gradients of the points.
            grad_threshold (float): Threshold for selecting points based on gradient.
            scene_extent (float): Extent of the scene to consider for densification.
            N (int, optional): Number of new points to generate for each selected point. Default is 2.

        Returns:
            None
        """
        n_init_centres = self.get_centre.shape[0] 
        # Extract centres that staify gradient condition
        padded_grad = torch.zeros((n_init_centres), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # Select centres with high positional gradients
        selected_centres_mask = torch.where(padded_grad >= grad_threshold, True, False) 
        # Select centres with high positional gradients and scaling factors
        selected_centres_mask = torch.logical_and(selected_centres_mask,
                                                  torch.max(self.get_scaling, dim=1).values > self.percent_dense * scaling_threshold)

        stds = self.get_scaling[selected_centres_mask].repeat(N, 1)  # Create N copies
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_centres_mask]).repeat(N,1,1) 
        new_centre = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_centre[selected_centres_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_centres_mask].repeat(N, 1) / (0.6 * N))
        new_rotation = self._rotation[selected_centres_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_centres_mask].repeat(N,1)
        new_emission_fc1_weights = self._emission_mlps.fc1_weights[selected_centres_mask].repeat(N, 1, 1)
        new_emission_fc1_bias = self._emission_mlps.fc1_bias[selected_centres_mask].repeat(N, 1)
        new_emission_fc2_weights = self._emission_mlps.fc2_weights[selected_centres_mask].repeat(N, 1, 1)
        new_emission_fc2_bias = self._emission_mlps.fc2_bias[selected_centres_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_centres_mask].repeat(N)
        
        self.densification_postfix(new_centre, new_scaling, new_rotation, new_opacity, new_emission_fc1_weights, new_emission_fc1_bias, new_emission_fc2_weights, new_emission_fc2_bias, new_tmp_radii)

        # Need to prune selected_centres after splitting
        prune_mask = torch.cat((selected_centres_mask, torch.zeros(N * selected_centres_mask.sum(), dtype=torch.bool, device="cuda")))
        self.prune_points(prune_mask)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_centres_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_centres_mask = torch.logical_and(selected_centres_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_centre = self._centre[selected_centres_mask]
        new_scaling = self._scaling[selected_centres_mask]
        new_rotation = self._rotation[selected_centres_mask]
        new_opacity = self._opacity[selected_centres_mask]
        new_emission_fc1_weights = self._emission_mlps.fc1_weights[selected_centres_mask]
        new_emission_fc1_bias = self._emission_mlps.fc1_bias[selected_centres_mask]
        new_emission_fc2_weights = self._emission_mlps.fc2_weights[selected_centres_mask]
        new_emission_fc2_bias = self._emission_mlps.fc2_bias[selected_centres_mask]

        new_tmp_radii = self.tmp_radii[selected_centres_mask]

        self.densification_postfix(new_centre, new_scaling, new_rotation, new_opacity, new_emission_fc1_weights, new_emission_fc1_bias, new_emission_fc2_weights, new_emission_fc2_bias, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,
                          scene_min=None, scene_max=None):
        grads = self.centre_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # Prune Gaussians whose centres escaped the scene bounding box
        if scene_min is not None and scene_max is not None:
            centres = self.get_centre.detach()
            out_of_bounds = (centres < scene_min).any(dim=1) | (centres > scene_max).any(dim=1)
            prune_mask = torch.logical_or(prune_mask, out_of_bounds)

        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        
        torch.cuda.empty_cache()

    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        Updates the densification statistics for the given viewspace points.

        Args:
            viewspace_point_tensor (torch.Tensor): A tensor containing the viewspace points.
            update_filter (torch.Tensor): A boolean tensor indicating which points to update.

        Updates:
            self.xyz_gradient_accum (torch.Tensor): Accumulates the gradient norms of the viewspace points.
            self.denom (torch.Tensor): Increments the count of updates for each point.
        """
        self.centre_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1   


class EmissionMLPs():

    def __init__(self, num_gaussians, input_size, hidden_size, output_size):
        
        # define MLP parameters as tensors
        self.fc1_weights = nn.Parameter(torch.randn((num_gaussians, hidden_size, input_size), device="cuda").contiguous().requires_grad_(True))
        self.fc1_bias = nn.Parameter(torch.randn((num_gaussians, hidden_size), device="cuda").contiguous().requires_grad_(True))

        self.fc2_weights = nn.Parameter(torch.randn((num_gaussians, output_size, hidden_size), device="cuda").contiguous().requires_grad_(True))
        self.fc2_bias = nn.Parameter(torch.randn((num_gaussians, output_size), device="cuda").contiguous().requires_grad_(True))

    def get_all_params(self):
        # flatten and concatenate all parameters into a single tensor
        return torch.cat([
            self.fc1_weights.view(-1),
            self.fc1_bias.view(-1),
            self.fc2_weights.view(-1),
            self.fc2_bias.view(-1),
        ]).contiguous()
    

class ConfidenceMLP(nn.Module):
    """
    Small MLP: rx_pos (3D) → scalar confidence  C > 1.

    Mirrors DUSt3R Eq. (4): the raw network output is passed through
    1 + exp(·) so that C > 1, which forces the network to always
    attempt reconstruction even for hard samples.

    Architecture:  Linear(3 → H) → ReLU → Linear(H → 1)
    """

    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        # Initialize fc2 bias so that initial confidence ≈ 1 + exp(0) = 2
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.to("cuda")

    def forward(self, rx_pos):
        """
        Args:
            rx_pos: (3,) or (B, 3) tensor – receiver position(s)
        Returns:
            confidence: scalar or (B,) tensor, always > 1
        """
        if rx_pos.dim() == 1:
            rx_pos = rx_pos.unsqueeze(0)  # (1, 3)
        raw = self.fc2(self.relu(self.fc1(rx_pos)))  # (B, 1)
        confidence = 1.0 + torch.exp(raw)             # C > 1, matching DUSt3R
        return confidence.squeeze(-1)                  # (B,) or scalar