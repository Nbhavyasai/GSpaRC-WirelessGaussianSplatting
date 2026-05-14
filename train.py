######Implement training loop###########
#   1. Load train data from preprocess.py
#   2. Load scene from gaussain.py
#   3. Plot initial scene and ground truth data - plotting.py
#   4. Initialize optimizer from optimizer.py
#   5. Start training loop
#         1. render scene from renderer.py
#         2. calculate loss from optimizer.py
#         3. update scene parameters with optimizer
#         4. update gaussians with update_gaussians.py
#   6. plot optimized scene and training curve - plotting.py
########################################

import os
import argparse
from shutil import copyfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.optim as optim
import numpy as np
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time  # Import time module for tracking training duration
from tqdm import tqdm  # Import tqdm
import sys  # Import sys for stdout/stderr redirection
import glob  # Import glob for finding checkpoints
os.environ['TENSORBOARD_PORT'] = '6007'
import csv

from preprocess_data import *
from initialization import *
from gaussian import *
from optimizer import *
from utils import *
from renderer import *
from plotting import *
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def circular_phase_l1_loss(phase_pred_norm, phase_gt_norm):
    """
    phase_pred_norm, phase_gt_norm: tensors in [0,1], e.g. shape (B, H, W)
    """
    diff = torch.abs(phase_pred_norm - phase_gt_norm)
    circ_diff = torch.minimum(diff, 1.0 - diff)
    return circ_diff.mean()   # L1 circular loss

def create_run_folder():
    """Create a new folder for this run with timestamp"""
    today_date = time.strftime("%m_%d") # Get current date as MM_DD
    base_results_folder = f"results_{today_date}" # Create base folder name like results_05_03
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_results_folder, f"run_{timestamp}") # Use the date-based folder
    os.makedirs(run_folder, exist_ok=True)
    return run_folder # Only return the folder path


def wireless_data_collate_fn(batch):
    """
    Custom collate function to handle WirelessData objects in the DataLoader.
    Converts a batch of WirelessData objects into a dictionary of tensors.
    """
    # Assuming WirelessData has attributes like 'spectrum_real' and others
    collated_batch = {
        'spectrum_real': torch.stack([item.spectrum_real for item in batch]),
        # Add other attributes as needed
    }
    return collated_batch


def calculate_psnr(pred, target):
    """Calculate PSNR between predicted and target images"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming normalized values between 0 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def calculate_rssi(signal):
    """Calculate RSSI in dBm from signal strength"""
    # Convert signal power to dBm
    # Assuming signal is power/magnitude, adding reference power of -30dBm
    power = torch.mean(signal ** 2)  # Average power
    rssi = 10 * torch.log10(power) - 30  # Convert to dBm with reference
    return rssi


def calculate_mse(pred, target):
    """Calculate Mean Square Error"""
    return torch.mean((pred - target) ** 2)


def calculate_mae(pred, target):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))


# Helper function for logging metric distributions to TensorBoard
def log_metric_distribution(writer, metric_name, data_tensor, iteration, unit=''):
    """Logs mean, median, std, histogram, and CDF plot for a given metric."""
    if not isinstance(data_tensor, torch.Tensor):
        data_tensor = torch.tensor(data_tensor)
        
    if data_tensor.numel() == 0:  # Handle empty tensors
        print(f"Warning: Empty data tensor for metric {metric_name} at iteration {iteration}. Skipping logging.")
        return

    mean_val = data_tensor.mean().item()
    median_val = data_tensor.median().item()
    std_val = data_tensor.std().item()

    writer.add_scalar(f'{metric_name}/mean', mean_val, iteration)
    writer.add_scalar(f'{metric_name}/median', median_val, iteration)
    writer.add_scalar(f'{metric_name}/std', std_val, iteration)
    writer.add_histogram(f'{metric_name}/distribution', data_tensor, iteration)

    # Create CDF plot
    sorted_data, _ = torch.sort(data_tensor)
    cdf = torch.linspace(0, 1, len(sorted_data))

    fig = plt.figure(figsize=(8, 6))
    plt.plot(sorted_data.cpu().numpy(), cdf.cpu().numpy(), label='CDF')
    plt.xlabel(f'{metric_name} Value {unit}')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{metric_name} Distribution (Iteration {iteration})')
    plt.grid(True)
    plt.legend()

    writer.add_figure(f'{metric_name}/cdf', fig, iteration)
    plt.close(fig)


class WGS_RFSPM_Runner:

    def __init__(self, mode, dataset_type, debug=False, num_test_plots=10, 
                 plot_initial_gaussians=False, plot_final_gaussians=False, # Add new arguments
                 checkpoint_interval=100000, checkpoint_dir="checkpoints", # Add checkpoint args
                 **kwargs): 
        # Create run folder at initialization
        self.run_folder = create_run_folder() # Get only the run folder path

        # Info from config yaml file
        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_train = kwargs['train']
        self.dataset_type = dataset_type
        self.debug = debug
        self.num_test_plots = num_test_plots # Store the argument
        self.plot_initial_gaussians = plot_initial_gaussians # Store flag
        self.plot_final_gaussians = plot_final_gaussians # Store flag
        self.checkpoint_interval = checkpoint_interval # Store checkpoint interval
        self.checkpoint_dir = os.path.join(self.run_folder, checkpoint_dir) # Create full checkpoint path
        os.makedirs(self.checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists

        ## Path settings
        self.expname = kwargs_path['expname']
        self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        os.makedirs(os.path.dirname(log_savepath), exist_ok=True)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='wgs_rfspm')
        self.logger.info("expname:%s, datadir:%s, logdir:%s", self.expname, self.datadir, self.logdir)
        self.logger.info(f"Run folder: {self.run_folder}")  # Log the run folder path
        
        # Change tensorboard writer to use run folder
        tensorboard_dir = os.path.join(self.run_folder, 'tensorboard')
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        ## Train Settings
        self.current_iteration = 1
        self.batch_size = 1
        self.total_iterations = kwargs_train['total_iterations']

        # Render Settings
        ## renderer is chosen and initialized here for the given dataset type ##
        self.scale_worldsize = kwargs_render['scale_worldsize']

        # Densification Logger
        if self.debug:
            self.densification_logger = open(os.path.join(self.logdir, 'densification_log.txt'), 'w')

        # Load dataset
        dataset = dataset_dict[dataset_type]                                        # choose dataset type
        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")
        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type=dataset_type)
        print("Loading training set...")
        self.train_set = dataset(self.datadir, train_index, self.scale_worldsize)
        print("Loading test set...")
        self.test_set = dataset(self.datadir, test_index, self.scale_worldsize)

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=wireless_data_collate_fn)
        print(f"Train set size: {len(self.train_set)}, Test set size: {len(self.test_set)}")

        # fixing the dimensions of scene for now
        self.xyz_min = np.array([-7.0, -4.0, 0.0])
        self.xyz_max = np.array([7.0, 6.0, 4.0])
        self.scene_extent = np.linalg.norm(self.xyz_max - self.xyz_min)

        # Optimization parameters instance
        self.optim_params = OptimizationParams()

        # Get optimizer parameters from kwargs
        optimizer_kwargs = kwargs.get('optimizer_params', {})
        self.optim_params = OptimizationParams(**optimizer_kwargs)
        
        # Log received optimizer parameters
        print("\nReceived optimizer parameters:")
        for key, value in optimizer_kwargs.items():
            print(f"{key}: {value}")

    def save_checkpoint(self, gaussian_model, iteration):
        """Saves the model and optimizer state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{iteration}.pth")
        
        # Gather state dictionaries - ensure keys match attributes used in load_checkpoint
        model_state = {
            '_centre': gaussian_model._centre,
            '_scaling': gaussian_model._scaling,
            '_rotation': gaussian_model._rotation,
            '_opacity': gaussian_model._opacity,
            'emission_fc1_weights': gaussian_model._emission_mlps.fc1_weights, # Match load_checkpoint keys
            'emission_fc1_bias': gaussian_model._emission_mlps.fc1_bias,       # Match load_checkpoint keys
            'emission_fc2_weights': gaussian_model._emission_mlps.fc2_weights, # Match load_checkpoint keys
            'emission_fc2_bias': gaussian_model._emission_mlps.fc2_bias,       # Match load_checkpoint keys
        }
        
        # Save confidence MLP state dict separately (it's an nn.Module)
        confidence_state = None
        if gaussian_model._confidence_mlp is not None:
            confidence_state = gaussian_model._confidence_mlp.state_dict()

        save_dict = {
            'iteration': iteration,
            'model_state_dict': model_state,
            'optimizer_state_dict': gaussian_model.optimizer.state_dict(),
            'confidence_mlp_state_dict': confidence_state,
        }
        
        torch.save(save_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, gaussian_model, checkpoint_path):
        """Loads the model and optimizer state from a checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        # Add weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=self.devices, weights_only=False) 
        
        # Load model state
        model_state = checkpoint['model_state_dict']
        # Directly assign loaded tensors to the model's parameters
        # Ensure the GaussianModel instance has these attributes initialized (even if empty)
        gaussian_model._centre = torch.nn.Parameter(model_state['_centre'].to(self.devices))
        gaussian_model._scaling = torch.nn.Parameter(model_state['_scaling'].to(self.devices))
        gaussian_model._rotation = torch.nn.Parameter(model_state['_rotation'].to(self.devices))
        gaussian_model._opacity = torch.nn.Parameter(model_state['_opacity'].to(self.devices))
        
        # Ensure the EmissionMLPs object exists and load its parameters
        if not hasattr(gaussian_model, '_emission_mlps') or gaussian_model._emission_mlps is None:
             # Initialize EmissionMLPs if it doesn't exist (adjust sizes if necessary)
             num_gaussians = model_state['_centre'].shape[0]
             gaussian_model._emission_mlps = EmissionMLPs(num_gaussians, 
                                                          gaussian_model.mlp_size[0], 
                                                          gaussian_model.mlp_size[1], 
                                                          gaussian_model.mlp_size[2])

        gaussian_model._emission_mlps.fc1_weights = torch.nn.Parameter(model_state['emission_fc1_weights'].to(self.devices))
        gaussian_model._emission_mlps.fc1_bias = torch.nn.Parameter(model_state['emission_fc1_bias'].to(self.devices))
        gaussian_model._emission_mlps.fc2_weights = torch.nn.Parameter(model_state['emission_fc2_weights'].to(self.devices))
        gaussian_model._emission_mlps.fc2_bias = torch.nn.Parameter(model_state['emission_fc2_bias'].to(self.devices))

        # Load confidence MLP state if available
        confidence_state = checkpoint.get('confidence_mlp_state_dict', None)
        if confidence_state is not None:
            hidden_size = getattr(self.optim_params, 'confidence_hidden_size', 32)
            if gaussian_model._confidence_mlp is None:
                from gaussian import ConfidenceMLP
                gaussian_model._confidence_mlp = ConfidenceMLP(hidden_size=hidden_size)
            gaussian_model._confidence_mlp.load_state_dict(confidence_state)
            gaussian_model._confidence_mlp.to(self.devices)
            print("Loaded confidence MLP state from checkpoint.")
        else:
            print("No confidence MLP state found in checkpoint (older checkpoint?).")

        # Load optimizer state
        # Important: Optimizer must be initialized *before* loading state_dict
        # training_setup should handle this.
        try:
            gaussian_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Optimizer state will be re-initialized.")


        iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from iteration {iteration} at {checkpoint_path}")
        return iteration

    def train(self):
        """train the model"""
        first_iter = 0
        gaussian_model = GaussianModel(logger=self.densification_logger, debug=self.debug)  # Pass logger here
        init_type = initializer_dict["uniform"]
        initializer = init_type(self.xyz_min, self.xyz_max, self.optim_params.cube_size)
        points = initializer.get_initial_points()

        # Log optimizer parameters
        print("\nOptimizer Parameters:")
        print("-" * 50)
        for param_name, param_value in vars(self.optim_params).items():
            if not param_name.startswith('_'):  # Only print public attributes
                print(f"{param_name}: {param_value}")
        print("-" * 50 + "\n")

        gaussian_model.initialize_gaussians(points)
        gaussian_model.training_setup(self.optim_params, spatial_lr_scale=self.scene_extent)

        # Print the number of initial Gaussians
        initial_gaussians = gaussian_model.get_centre.size(0)
        print(f"Number of initial Gaussians: {initial_gaussians}")
        # Log initial number of Gaussians to TensorBoard
        self.writer.add_scalar('Gaussians/count', initial_gaussians, 0)  # Use iteration 0 for initial count

        # Plot the initial Gaussians if requested
        if self.plot_initial_gaussians:
            print("Plotting initial Gaussians...")
            plot_initialization(self.xyz_min, self.xyz_max, gaussian_model, 
                              save_path=os.path.join(self.run_folder, "initial_gaussians.png"))

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        wireless_data_train = self.train_set.get_data().copy()
        wireless_data_indices = list(range(len(wireless_data_train)))
        rays_directions = self.train_set.ray_directions

        # Track start time
        start_time = time.time()

        # Number of initial Gaussians
        initial_gaussians = gaussian_model.get_centre.size(0)

        # Create the train subfolder for plots once before the loop starts
        train_plots_folder = os.path.join(self.run_folder, "train")
        os.makedirs(train_plots_folder, exist_ok=True)

        # Initialize progress bar for training
        progress_bar = tqdm(range(first_iter, self.optim_params.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.optim_params.iterations + 1):

            # Reload training data if all samples are used
            if not wireless_data_train:
                wireless_data_train = self.train_set.get_data().copy()
                np.random.shuffle(wireless_data_train)
            wireless_data = wireless_data_train.pop()

            # Update learning rate for the current iteration
            gaussian_model.update_learning_rate(iteration)

            # Ensure the number of Gaussians does not exceed the safe limit
            max_gaussians = 2**31 - 1  # Adjust based on the data type used in rasterizer
            if gaussian_model.get_num_gaussians > max_gaussians:
                raise RuntimeError(f"Number of Gaussians ({gaussian_model.get_num_gaussians}) exceeds the safe limit ({max_gaussians}).")

            # Render the scene using the current Gaussian model and training data
            render_start = time.time()
            rendered_pkg = render(wireless_data, gaussian_model)
            render_end = time.time()
            render_time = render_end - render_start

            rendered_signal_real, rendered_signal_imag = rendered_pkg['render']
            visibility_filter, viewspace_point_tensor, radii = (
                rendered_pkg["visibility_filter"], 
                rendered_pkg["viewspace_points"], 
                rendered_pkg["radii"]
            )
            # Save spectrum comparison plot every 10000 iterations
            if (iteration % 10000 == 0) :
                output_img = rendered_signal_real.detach().cpu().numpy()
                gt_img = wireless_data.spectrum_real.cpu().numpy()
                

                # Plot side by side
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(gt_img, cmap='gray')
                plt.title('Ground Truth Spectrum')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(output_img, cmap='gray')
                plt.title('Predicted Spectrum')
                plt.axis('off')

                plt.tight_layout()
                # Save the plot inside the 'train' subfolder with updated name
                plot_save_path = os.path.join(train_plots_folder, f"spectrum_iter_{iteration}.png") # Rename here
                plt.savefig(plot_save_path)
                plt.close() # Close the figure to free memory

            # Calculate loss using L1 loss and SSIM
            gt_signal_real = wireless_data.spectrum_real.cuda()
            
            loss_l1 = l1_loss(rendered_signal_real, gt_signal_real)

            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(rendered_signal_real.unsqueeze(0).unsqueeze(0), gt_signal_real.unsqueeze(0).unsqueeze(0))
            else:
                ssim_value = ssim(rendered_signal_real, gt_signal_real)

            reconstruction_loss = (1.0 - self.optim_params.lambda1) * loss_l1 + self.optim_params.lambda1 * (1.0 - ssim_value)

            # ── DUSt3R-style confidence-weighted loss (Eq. 4) ──
            # L_conf = C * L_reconstruction - alpha * log(C)
            # C > 1 by construction (1 + exp(raw)), so log(C) > 0.
            # The network can down-weight hard samples (low C) but pays a
            # penalty (-alpha * log C becomes more negative → larger loss).
            confidence = rendered_pkg["confidence"]  # scalar, > 1
            alpha = self.optim_params.confidence_alpha
            loss = confidence * reconstruction_loss - alpha * torch.log(confidence)

            # Perform backpropagation to compute gradients
            loss.backward()

            with torch.no_grad():
                # Check gradients for NaN or Inf, clip gradients if necessary
                if torch.isnan(gaussian_model.get_centre.grad).any() or torch.isinf(gaussian_model.get_centre.grad).any():
                    nan_mask = torch.isnan(gaussian_model.get_centre.grad)
                    gaussian_model._centre.grad[nan_mask] = torch.zeros_like(gaussian_model.get_centre.grad[nan_mask])  

                # Log loss to TensorBoard and update progress bar
                self.writer.add_scalar('Loss/train', loss.item(), iteration)
                self.writer.add_scalar('Confidence/train_value', confidence.item(), iteration)
                self.writer.add_scalar('Loss/train_reconstruction', reconstruction_loss.item(), iteration)
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss.item():.7f}", "Conf": f"{confidence.item():.3f}"})
                    progress_bar.update(10)
                if iteration == self.optim_params.iterations:
                    progress_bar.close()

                # Report metrics from test data every 1000 iterations
                if iteration % 1000 == 0:
                    total_test_loss = 0.0
                    total_test_samples = 0
                    psnr_values = []
                    ssim_values = []
                    rssi_diff_values = []
                    mse_values = []
                    confidence_values = []

                    for test_data in self.test_set.get_data():
                        rendered_pkg = render(test_data, gaussian_model)
                        rendered_signal_real, _ = rendered_pkg['render']
                        gt_signal_real = test_data.spectrum_real.cuda()

                        # Calculate metrics
                        psnr_values.append(calculate_psnr(rendered_signal_real, gt_signal_real).item())
                        rssi_diff_values.append((calculate_rssi(rendered_signal_real) - calculate_rssi(gt_signal_real)).item())
                        mse_values.append(calculate_mse(rendered_signal_real, gt_signal_real).item())
                        confidence_values.append(rendered_pkg["confidence"].item())

                        if FUSED_SSIM_AVAILABLE:
                            ssim_value = fused_ssim(rendered_signal_real.unsqueeze(0).unsqueeze(0),
                                                  gt_signal_real.unsqueeze(0).unsqueeze(0))
                        else:
                            ssim_value = ssim(rendered_signal_real, gt_signal_real)
                        ssim_values.append(ssim_value.item())

                        # Calculate test loss for this sample
                        loss_l1_test = l1_loss(rendered_signal_real, gt_signal_real)
                        loss_test = (1.0 - self.optim_params.lambda1) * loss_l1_test + self.optim_params.lambda1 * (1.0 - ssim_value)
                        total_test_loss += loss_test.item()
                        total_test_samples += 1

                    # Log average test loss
                    avg_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
                    self.writer.add_scalar('Loss/test', avg_test_loss, iteration)

                    # Log distributions using the helper function
                    log_metric_distribution(self.writer, 'PSNR', psnr_values, iteration, unit=' (dB)')
                    log_metric_distribution(self.writer, 'SSIM', ssim_values, iteration)
                    log_metric_distribution(self.writer, 'RSSI_Difference', rssi_diff_values, iteration, unit=' (dBm)')
                    log_metric_distribution(self.writer, 'MSE', mse_values, iteration)
                    log_metric_distribution(self.writer, 'Confidence_test', confidence_values, iteration)

                # ── Densification & pruning ──
                # Precompute scene bounds tensors (with margin) for OOB pruning and clamping.
                bounds_margin = 4.0
                scene_min_t = torch.tensor(self.xyz_min - bounds_margin, device='cuda', dtype=torch.float32)
                scene_max_t = torch.tensor(self.xyz_max + bounds_margin, device='cuda', dtype=torch.float32)

                if iteration < self.optim_params.densify_until_iter:
                    # Keep track of max screen-space radius per Gaussian (for big-blob pruning)
                    if visibility_filter.numel() > 0:
                        vis_idx = visibility_filter.squeeze(-1) if visibility_filter.dim() > 1 else visibility_filter
                        gaussian_model.max_radii2D[vis_idx] = torch.max(
                            gaussian_model.max_radii2D[vis_idx], radii[vis_idx].float()
                        )
                        gaussian_model.add_densification_stats(viewspace_point_tensor, vis_idx)
                        if iteration % 100 == 0:
                            self.writer.add_scalar('Densify/visible_count', vis_idx.sum().item(), iteration)
                            self.writer.add_scalar('Densify/grad_mean', viewspace_point_tensor.grad[vis_idx, :2].abs().mean().item(), iteration)

                    if (iteration > self.optim_params.densify_from_iter
                            and iteration % self.optim_params.densification_interval == 0):
                        size_threshold = 20 if iteration > self.optim_params.opacity_reset_interval else None
                        gaussian_model.densify_and_prune(
                            max_grad      = self.optim_params.densify_grad_threshold,
                            min_opacity   = 0.01,
                            extent        = self.scene_extent,
                            max_screen_size = size_threshold,
                            radii         = radii,
                            scene_min     = scene_min_t,
                            scene_max     = scene_max_t,
                        )
                        self.writer.add_scalar('Gaussians/count', gaussian_model.get_num_gaussians, iteration)

                    if (iteration % self.optim_params.opacity_reset_interval == 0
                            and iteration > 0
                            and iteration < self.optim_params.densify_until_iter):
                        gaussian_model.reset_opacity()

                # ── Post-densification pruning (runs even after densify_until_iter) ──
                # Periodically kill giant / escaped / near-transparent Gaussians
                # so they can't grow unchecked in the pure-optimization phase.
                elif iteration % 10000 == 0:
                    with torch.no_grad():
                        prune_mask = (gaussian_model.get_opacity < 0.001).squeeze()
                        big_ws = gaussian_model.get_scaling.max(dim=1).values > 0.1 * self.scene_extent
                        centres = gaussian_model.get_centre.detach()
                        oob = (centres < scene_min_t).any(dim=1) | (centres > scene_max_t).any(dim=1)
                        prune_mask = prune_mask | big_ws | oob
                        if prune_mask.any():
                            gaussian_model.tmp_radii = radii
                            gaussian_model.prune_points(prune_mask)
                            gaussian_model.tmp_radii = None
                            self.writer.add_scalar('Gaussians/count', gaussian_model.get_num_gaussians, iteration)

                # Optimizer step
                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)

                # Clamp Gaussian centres to scene bounding box (with margin)
                # to prevent them from escaping to distant positions.
                with torch.no_grad():
                    gaussian_model._centre.data.clamp_(min=scene_min_t, max=scene_max_t)

                # Check Gaussian centers for NaN after optimizer step
                if torch.isnan(gaussian_model.get_centre).any():
                    raise RuntimeError("NaN detected in Gaussian centers after optimizer step.")
                
                # Save checkpoint
                if self.checkpoint_interval > 0 and iteration % self.checkpoint_interval == 0:
                    self.save_checkpoint(gaussian_model, iteration)

        # Track end time and calculate training duration
        end_time = time.time()
        training_duration = end_time - start_time

        # Number of final Gaussians after training
        final_gaussians = gaussian_model.get_centre.size(0)
        print(f"Number of final Gaussians after training: {final_gaussians}")

        # Print total training duration
        print(f"Time taken for training: {training_duration / 60:.2f} minutes")

        # Plot final Gaussians in the run folder if requested
        if self.plot_final_gaussians:
            print("Plotting final Gaussians...")
            plot_final_gaussians(self.xyz_min, self.xyz_max, gaussian_model,
                               save_path=os.path.join(self.run_folder, "final_gaussians.png"))

        # --- Automatically run evaluation after training ---
        print("\n--- Starting Post-Training Evaluation ---")
        latest_checkpoint_path = None
        if self.checkpoint_interval > 0 and os.path.exists(self.checkpoint_dir):
            # Find the latest checkpoint
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_iter_*.pth"))
            if checkpoint_files:
                iterations = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
                latest_iteration = max(iterations)
                latest_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{latest_iteration}.pth")
                print(f"Found latest checkpoint: {latest_checkpoint_path}")
            else:
                print("No checkpoint files found in the checkpoint directory.")
        else:
            print("Checkpoint saving was disabled or directory doesn't exist. Skipping evaluation.")

        if latest_checkpoint_path:
            # Run evaluation using the latest checkpoint
            self.evaluate(latest_checkpoint_path)
        else:
            print("Could not find a checkpoint to load for evaluation.")
        print("--- Post-Training Evaluation Finished ---")


    def evaluate(self, checkpoint_path):
        """Load a checkpoint and evaluate the model on the test set."""
        print(f"Starting evaluation using checkpoint: {checkpoint_path}")

        # Initialize Gaussian Model
        gaussian_model = GaussianModel(logger=None, debug=False)

        # Load checkpoint FIRST
        loaded_iteration = self.load_checkpoint(gaussian_model, checkpoint_path)

        # Setup optimizer after loading
        gaussian_model.training_setup(self.optim_params, spatial_lr_scale=self.scene_extent)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.devices, weights_only=False)
            gaussian_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully reloaded optimizer state after setup.")
        except Exception as e:
            print(f"Warning: Could not reload optimizer state after setup: {e}. Optimizer will use initial state.")

        # Create evaluation directory
        eval_plots_folder = os.path.join(self.run_folder, "evaluation")
        os.makedirs(eval_plots_folder, exist_ok=True)
        print(f"Saving evaluation plots to: {eval_plots_folder}")

        def evaluate_split(split_name, dataset_obj):
            """
            Evaluate a given dataset split ('test' or 'train').
            Returns summary metrics and per-sample rows.
            """
            print(f"\n--- Evaluating {split_name.upper()} split ---")

            # Create subfolder for per-sample spectrum plots (same style as training plots)
            split_plots_folder = os.path.join(eval_plots_folder, f"{split_name}_plots")
            os.makedirs(split_plots_folder, exist_ok=True)
            print(f"Saving per-sample {split_name} spectrum plots to: {split_plots_folder}")

            total_loss = 0.0
            total_samples = 0

            psnr_values = []
            ssim_values = []
            rssi_diff_values = []
            mae_values = []
            render_times = []
            confidence_values = []

            per_sample_rows = []
            all_rendered = []

            data_list = dataset_obj.get_data()

            with torch.no_grad():
                progress_bar = tqdm(
                    enumerate(data_list),
                    total=len(data_list),
                    desc=f"Evaluating {split_name}"
                )

                for idx, sample_data in progress_bar:
                    render_start = time.time()
                    rendered_pkg = render(sample_data, gaussian_model)
                    render_end = time.time()

                    current_render_time_ms = (render_end - render_start) * 1000.0
                    render_times.append(render_end - render_start)

                    rendered_signal_real, _ = rendered_pkg['render']
                    gt_signal_real = sample_data.spectrum_real.cuda()

                    # Metrics
                    psnr_val = calculate_psnr(rendered_signal_real, gt_signal_real).item()
                    rssi_diff_val = (calculate_rssi(rendered_signal_real) - calculate_rssi(gt_signal_real)).item()
                    mae_val = calculate_mae(rendered_signal_real, gt_signal_real).item()
                    ssim_val = ssim(rendered_signal_real, gt_signal_real).item()
                    conf_val = rendered_pkg["confidence"].item()

                    # Loss
                    loss_l1 = l1_loss(rendered_signal_real, gt_signal_real)
                    loss_val = (1.0 - self.optim_params.lambda1) * loss_l1 + self.optim_params.lambda1 * (1.0 - ssim_val)

                    total_loss += loss_val.item()
                    total_samples += 1

                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)
                    rssi_diff_values.append(rssi_diff_val)
                    mae_values.append(mae_val)
                    confidence_values.append(conf_val)

                    rendered_signal_np = rendered_signal_real.detach().cpu().numpy()
                    all_rendered.append(rendered_signal_np.copy())

                    sample_id = dataset_obj.dataset_index[idx]

                    # ── Save side-by-side GT vs Predicted spectrum plot (same style as training) ──
                    gt_img = gt_signal_real.detach().cpu().numpy()
                    output_img = rendered_signal_np

                    plt.figure(figsize=(12, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(gt_img, cmap='gray')
                    plt.title('Ground Truth Spectrum')
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(output_img, cmap='gray')
                    plt.title('Predicted Spectrum')
                    plt.axis('off')

                    plt.suptitle(
                        f'{split_name.upper()} | idx={idx} id={sample_id} | '
                        f'PSNR={psnr_val:.2f}  SSIM={ssim_val:.3f}  MAE={mae_val:.4f}  Conf={conf_val:.3f}',
                        fontsize=11
                    )
                    plt.tight_layout()

                    sample_plot_path = os.path.join(
                        split_plots_folder,
                        f"spectrum_{split_name}_idx{idx}_id{sample_id}.png"
                    )
                    plt.savefig(sample_plot_path, dpi=120)
                    plt.close()

                    per_sample_rows.append({
                        "sample_idx": idx,
                        "sample_id": sample_id,
                        "split": split_name,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "mae": mae_val,
                        "rssi_diff": rssi_diff_val,
                        "render_time_ms": current_render_time_ms,
                        "loss": loss_val.item(),
                        "confidence": conf_val,
                    })

                    # Print to tqdm bar
                    progress_bar.set_postfix({
                        "SampleIdx": idx,
                        "ID": sample_id,
                        "PSNR": f"{psnr_val:.2f}",
                        "SSIM": f"{ssim_val:.3f}",
                        "MAE": f"{mae_val:.4f}",
                        "Conf": f"{conf_val:.3f}",
                        "RenderTime(ms)": f"{current_render_time_ms:.2f}"
                    })

                    # Also print an explicit log line for parsing later
                    print(
                        f"[{split_name.upper()}] "
                        f"SampleIdx={idx} "
                        f"ID={sample_id} "
                        f"PSNR={psnr_val:.4f} "
                        f"SSIM={ssim_val:.6f} "
                        f"MAE={mae_val:.6f} "
                        f"RSSI_DIFF={rssi_diff_val:.6f} "
                        f"Confidence={conf_val:.6f} "
                        f"RenderTimeMS={current_render_time_ms:.4f} "
                        f"LOSS={loss_val.item():.6f}"
                    )

            # Save rendered outputs
            all_rendered_np = np.stack(all_rendered, axis=0)
            rendered_path = os.path.join(eval_plots_folder, f"gs_output_real_{split_name}.npy")
            np.save(rendered_path, all_rendered_np)
            print(f"Saved {split_name} rendered outputs to: {rendered_path} with shape {all_rendered_np.shape}")

            # Save per-sample CSV
            metrics_csv_path = os.path.join(eval_plots_folder, f"per_sample_metrics_{split_name}.csv")
            with open(metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample_idx",
                    "sample_id",
                    "split",
                    "psnr",
                    "ssim",
                    "mae",
                    "rssi_diff",
                    "render_time_ms",
                    "loss",
                    "confidence",
                ])
                for row in per_sample_rows:
                    writer.writerow([
                        row["sample_idx"],
                        row["sample_id"],
                        row["split"],
                        row["psnr"],
                        row["ssim"],
                        row["mae"],
                        row["rssi_diff"],
                        row["render_time_ms"],
                        row["loss"],
                        row["confidence"],
                    ])
            print(f"Saved per-sample {split_name} metrics to: {metrics_csv_path}")

            # ── Confidence vs Loss correlation plot ──
            if confidence_values and len(confidence_values) > 1:
                conf_arr = np.array(confidence_values)
                loss_arr = np.array([r["loss"] for r in per_sample_rows])
                mae_arr  = np.array(mae_values)
                sample_ids = [r["sample_id"] for r in per_sample_rows]

                # --- Plot 1: Scatter plot with trend line ---
                fig, ax1 = plt.subplots(figsize=(10, 7))

                scatter = ax1.scatter(conf_arr, loss_arr, c=mae_arr, cmap='RdYlGn_r',
                                      alpha=0.7, edgecolors='k', linewidths=0.3, s=40)
                cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
                cbar.set_label('MAE', fontsize=11)

                # Trend line (linear fit)
                if np.std(conf_arr) > 1e-8:  # avoid degenerate case
                    z = np.polyfit(conf_arr, loss_arr, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(conf_arr.min(), conf_arr.max(), 100)
                    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')

                    # Pearson correlation
                    corr = np.corrcoef(conf_arr, loss_arr)[0, 1]
                    ax1.set_title(f'{split_name.upper()} — Confidence vs Loss  (Pearson r = {corr:.3f})',
                                  fontsize=13)
                    ax1.legend(fontsize=10)
                else:
                    ax1.set_title(f'{split_name.upper()} — Confidence vs Loss  (constant confidence)',
                                  fontsize=13)

                ax1.set_xlabel('Learned Confidence (C)', fontsize=12)
                ax1.set_ylabel('Reconstruction Loss', fontsize=12)
                ax1.grid(True, alpha=0.3)
                fig.tight_layout()
                scatter_path = os.path.join(eval_plots_folder,
                                            f"confidence_vs_loss_scatter_{split_name}.png")
                fig.savefig(scatter_path, dpi=150)
                plt.close(fig)
                print(f"Saved confidence-vs-loss scatter to: {scatter_path}")

                # --- Plot 2: Dual-axis sorted bar chart ---
                sort_idx = np.argsort(loss_arr)[::-1]  # highest loss first
                sorted_loss = loss_arr[sort_idx]
                sorted_conf = conf_arr[sort_idx]
                sorted_ids  = [sample_ids[i] for i in sort_idx]
                x_pos = np.arange(len(sorted_loss))

                fig2, ax_left = plt.subplots(figsize=(max(12, len(sorted_loss) * 0.18), 6))
                ax_right = ax_left.twinx()

                bar_width = 0.4
                ax_left.bar(x_pos - bar_width/2, sorted_loss, bar_width,
                            color='#e74c3c', alpha=0.75, label='Loss')
                ax_right.bar(x_pos + bar_width/2, sorted_conf, bar_width,
                             color='#2ecc71', alpha=0.75, label='Confidence')

                ax_left.set_xlabel('Samples (sorted by loss, descending)', fontsize=11)
                ax_left.set_ylabel('Reconstruction Loss', fontsize=11, color='#e74c3c')
                ax_right.set_ylabel('Confidence (C)', fontsize=11, color='#2ecc71')
                ax_left.tick_params(axis='y', labelcolor='#e74c3c')
                ax_right.tick_params(axis='y', labelcolor='#2ecc71')

                # Show sample IDs on x-axis if not too many
                if len(sorted_ids) <= 80:
                    ax_left.set_xticks(x_pos)
                    ax_left.set_xticklabels(sorted_ids, rotation=90, fontsize=6)
                else:
                    ax_left.set_xticks([])

                fig2.suptitle(f'{split_name.upper()} — Loss & Confidence per Sample (sorted by loss)',
                              fontsize=13)
                lines_left, labels_left = ax_left.get_legend_handles_labels()
                lines_right, labels_right = ax_right.get_legend_handles_labels()
                ax_left.legend(lines_left + lines_right, labels_left + labels_right,
                               loc='upper right', fontsize=10)
                fig2.tight_layout()
                bar_path = os.path.join(eval_plots_folder,
                                        f"confidence_vs_loss_bars_{split_name}.png")
                fig2.savefig(bar_path, dpi=150)
                plt.close(fig2)
                print(f"Saved confidence-vs-loss bar chart to: {bar_path}")

            # Summary
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
            avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
            avg_rssi_diff = np.mean(rssi_diff_values) if rssi_diff_values else 0.0
            avg_mae = np.mean(mae_values) if mae_values else 0.0
            avg_render_time = np.mean(render_times) * 1000.0 if render_times else 0.0
            avg_confidence = np.mean(confidence_values) if confidence_values else 0.0

            print(f"\n--- {split_name.upper()} Evaluation Results ---")
            print(f"Checkpoint Iteration: {loaded_iteration}")
            print(f"Average Loss: {avg_loss:.7f}")
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average RSSI Difference: {avg_rssi_diff:.2f} dBm")
            print(f"Average MAE: {avg_mae:.7f}")
            print(f"Average Confidence: {avg_confidence:.4f}")
            print(f"Average Render Time: {avg_render_time:.2f} ms")
            print(f"-----------------------------------\n")

            return {
                "avg_loss": avg_loss,
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "avg_rssi_diff": avg_rssi_diff,
                "avg_mae": avg_mae,
                "avg_confidence": avg_confidence,
                "avg_render_time": avg_render_time,
                "psnr_values": psnr_values,
                "ssim_values": ssim_values,
                "rssi_diff_values": rssi_diff_values,
                "mae_values": mae_values,
                "confidence_values": confidence_values,
                "per_sample_rows": per_sample_rows,
            }

        # Evaluate test split only
        test_results = evaluate_split("test", self.test_set)

        # Log summary metrics to TensorBoard
        self.writer.add_scalar('Loss/eval_test', test_results["avg_loss"], loaded_iteration)

        self.writer.add_scalar('Time/avg_render_test_ms', test_results["avg_render_time"], loaded_iteration)

        log_metric_distribution(self.writer, 'PSNR_test_eval', test_results["psnr_values"], loaded_iteration, unit=' (dB)')
        log_metric_distribution(self.writer, 'SSIM_test_eval', test_results["ssim_values"], loaded_iteration)
        log_metric_distribution(self.writer, 'RSSI_Difference_test_eval', test_results["rssi_diff_values"], loaded_iteration, unit=' (dBm)')
        log_metric_distribution(self.writer, 'MAE_test_eval', test_results["mae_values"], loaded_iteration)
        log_metric_distribution(self.writer, 'Confidence_test_eval', test_results["confidence_values"], loaded_iteration)

        print("Evaluation finished.")


    def __del__(self):
        # Close TensorBoard writer
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()
        # Close densification logger if it exists
        if hasattr(self, 'densification_logger') and self.densification_logger and not self.densification_logger.closed:
            self.densification_logger.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mimo-csi.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Execution mode: train or eval')
    parser.add_argument('--dataset_type', type=str, default='mimo')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for densification and pruning')
    parser.add_argument('--num_test_plots', type=int, default=10, help='Number of test samples to plot after training (only used in train mode)')
    parser.add_argument('--plot_initial_gaussians', action='store_true', default=False, help='Plot the initial distribution of Gaussians')
    parser.add_argument('--plot_final_gaussians', action='store_true', default=False, help='Plot the final distribution of Gaussians after training')
    parser.add_argument('--checkpoint_interval', type=int, default=100000, help='Interval for saving model checkpoints (train mode)')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Subdirectory for saving/loading checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint file to load for evaluation (eval mode)')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    # Validate arguments based on mode
    if args.mode == 'eval' and args.load_checkpoint is None:
        parser.error("--load_checkpoint is required when --mode is 'eval'")
    if args.mode == 'train' and args.load_checkpoint is not None:
        print("Warning: --load_checkpoint is provided but --mode is 'train'. Checkpoint loading is only implemented for 'eval' mode. Training will start from scratch.")


    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()

    # Add checkpoint args from parser to kwargs if they exist, otherwise use defaults
    checkpoint_args = {
        'checkpoint_interval': args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 10000,
        'checkpoint_dir': args.checkpoint_dir if hasattr(args, 'checkpoint_dir') else "checkpoints"
    }

    # Create the worker instance FIRST to establish the run_folder
    worker = WGS_RFSPM_Runner(mode=args.mode, 
                              dataset_type=args.dataset_type, 
                              debug=args.debug, 
                              num_test_plots=args.num_test_plots, 
                              plot_initial_gaussians=args.plot_initial_gaussians, # Pass argument
                              plot_final_gaussians=args.plot_final_gaussians, # Pass argument
                              **checkpoint_args, # Pass checkpoint arguments
                              **kwargs) 

    # Store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handle = None # Initialize handle to None

    try:
        # Redirect stdout and stderr AFTER run_folder is created
        output_log_path = os.path.join(worker.run_folder, 'output.log')
        print(f"Redirecting stdout and stderr to: {output_log_path}")  # Print to original stdout before redirection
        try:
            # Open the file in append mode, buffering=1 means line buffered
            log_file_handle = open(output_log_path, 'a', buffering=1)
            sys.stdout = log_file_handle
            sys.stderr = log_file_handle
        except Exception as e:
            # If redirection fails, print error to original stderr and continue
            print(f"Error redirecting stdout/stderr: {e}", file=original_stderr)
            # Keep original streams if redirection failed
            sys.stdout = original_stdout
            sys.stderr = original_stderr


        ## backup config file
        if args.mode == 'train':
            copyfile(args.config, os.path.join(worker.run_folder, 'config.yml'))

        # Now run the training or testing
        if args.mode == 'train':
            print("Starting training...")  # This will go to output.log
            worker.train()
            print("Training finished.")  # This will go to output.log
        elif args.mode == 'test':
            print("Starting testing...")  # This will go to output.log
            if args.dataset_type == 'rfid':
                worker.eval_network_spectrum()
            elif args.dataset_type == 'ble':
                worker.eval_network_rssi()
            elif args.dataset_type == 'mimo':
                worker.eval_network_csi()
            print("Testing finished.")  # This will go to output.log
        elif args.mode == 'eval': # Add evaluation mode execution
            print("Starting evaluation...") # This will go to output.log
            worker.evaluate(args.load_checkpoint)

    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Close the log file handle if it was opened
        if log_file_handle is not None:
            print(f"Closing log file: {output_log_path}") # Optional: Log closing action to original stdout
            log_file_handle.close()
        # Explicitly delete worker to trigger __del__ before script exits (optional but good practice)
        del worker