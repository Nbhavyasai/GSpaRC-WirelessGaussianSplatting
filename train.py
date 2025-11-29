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


# Add at the beginning of the file, after imports
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


# Add after other utility functions
def calculate_rssi(signal):
    """Calculate RSSI in dBm from signal strength"""
    # Convert signal power to dBm
    # Assuming signal is power/magnitude, adding reference power of -30dBm
    power = torch.mean(signal ** 2)  # Average power
    rssi = 10 * torch.log10(power) - 30  # Convert to dBm with reference
    return rssi


# Add after other utility functions
def calculate_mse(pred, target):
    """Calculate Mean Square Error"""
    return torch.mean((pred - target) ** 2)


# Add new function for MAE
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
        self.xyz_min = np.array([-30.0, -30.0, 0.0])
        self.xyz_max = np.array([30.0, 30.0, 30.0])

        # # optimization parameters instance
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
        
        save_dict = {
            'iteration': iteration,
            'model_state_dict': model_state,
            'optimizer_state_dict': gaussian_model.optimizer.state_dict(),
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

    def _find_latest_checkpoint(self):
        """Return the path to the latest checkpoint in self.checkpoint_dir, or None."""
        if not (self.checkpoint_interval > 0 and os.path.exists(self.checkpoint_dir)):
            return None
        files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_iter_*.pth"))
        if not files:
            return None
        iters = []
        for f in files:
            try:
                iters.append(int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
            except Exception:
                iters.append(-1)
        if max(iters) < 0:
            return None
        return files[int(np.argmax(iters))]

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

        gaussian_model.training_setup(self.optim_params)

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
            # loss_ssim = ssim(rendered_signal_real, gt_signal_real)
            # loss = (1-self.optim_params.lambda1) * loss_l1 + self.optim_params.lambda1 * loss_ssim

            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(rendered_signal_real.unsqueeze(0).unsqueeze(0), gt_signal_real.unsqueeze(0).unsqueeze(0))
            else:
                ssim_value = ssim(rendered_signal_real, gt_signal_real)

            loss = (1.0 - self.optim_params.lambda_ssim) * loss_l1 + self.optim_params.lambda_ssim * (1.0 - ssim_value)

            # Perform backpropagation to compute gradients
            loss.backward()

            with torch.no_grad():
                # Check gradients for NaN or Inf, clip gradients if necessary
                if torch.isnan(gaussian_model.get_centre.grad).any() or torch.isinf(gaussian_model.get_centre.grad).any():
                    nan_mask = torch.isnan(gaussian_model.get_centre.grad)
                    gaussian_model._centre.grad[nan_mask] = torch.zeros_like(gaussian_model.get_centre.grad[nan_mask])  

                # Log loss to TensorBoard and update progress bar
                self.writer.add_scalar('Loss/train', loss.item(), iteration)
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
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

                    for test_data in self.test_set.get_data():
                        # ... (rendering test data) ...
                        rendered_pkg = render(test_data, gaussian_model)
                        rendered_signal_real, _ = rendered_pkg['render']
                        gt_signal_real = test_data.spectrum_real.cuda()

                        # Calculate metrics
                        psnr_values.append(calculate_psnr(rendered_signal_real, gt_signal_real).item())
                        rssi_diff_values.append((calculate_rssi(rendered_signal_real) - calculate_rssi(gt_signal_real)).item())
                        mse_values.append(calculate_mse(rendered_signal_real, gt_signal_real).item())

                        if FUSED_SSIM_AVAILABLE:
                            ssim_value = fused_ssim(rendered_signal_real.unsqueeze(0).unsqueeze(0),
                                                  gt_signal_real.unsqueeze(0).unsqueeze(0))
                        else:
                            ssim_value = ssim(rendered_signal_real, gt_signal_real)
                        ssim_values.append(ssim_value.item())

                        # Calculate test loss for this sample
                        loss_l1_test = l1_loss(rendered_signal_real, gt_signal_real)
                        loss_test = (1.0 - self.optim_params.lambda_ssim) * loss_l1_test + self.optim_params.lambda_ssim * (1.0 - ssim_value)
                        total_test_loss += loss_test.item()
                        total_test_samples += 1

                    # Log average test loss
                    avg_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
                    self.writer.add_scalar('Loss/test', avg_test_loss, iteration)

                    # Log distributions using the helper function
                    # log_metric_distribution(self.writer, 'PSNR', psnr_values, iteration, unit=' (dB)')
                    log_metric_distribution(self.writer, 'SSIM', ssim_values, iteration)
                    # log_metric_distribution(self.writer, 'RSSI_Difference', rssi_diff_values, iteration, unit=' (dBm)')
                    # log_metric_distribution(self.writer, 'MSE', mse_values, iteration)

                # if iteration < self.optim_params.densify_until_iter:
                #     # Keep track of max radii in image-space for pruning
                #     gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration > self.optim_params.densify_from_iter and iteration % self.optim_params.densification_interval == 0:
                #         print(f"Radii tensor before: {radii.shape}, {radii.device}")
                #         size_threshold = 20 if iteration > self.optim_params.opacity_reset_interval else None
                #         gaussian_model.densify_and_prune(self.optim_params.densify_grad_threshold, 0.01, 8.0, size_threshold, radii)
                #         # print(f"Radii tensor after: {radii.shape}, {radii.device}")
                #         new_num_gauss = gaussian_model.get_num_gaussians
                #         print(f"densification and pruning complete")
                #         # Update TensorBoard with counts specifically for densification/pruning
                #         self.writer.add_scalar('Gaussians/count', new_num_gauss, iteration)

                #     if iteration % self.optim_params.opacity_reset_interval == 0:
                #         gaussian_model.reset_opacity()

                # Update optimizer step logic
                if iteration < self.optim_params.iterations:
                    gaussian_model.optimizer.step()
                    gaussian_model.optimizer.zero_grad(set_to_none=True)

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

        
        # --- Post-Training Evaluation ---
        print("\n--- Post-Training Evaluation ---")
        choice = getattr(self, "eval_after_train", "model")
        if choice == 'off':
            print("Skipping evaluation after training (eval_after_train=off).")
        elif choice == 'model':
            # Evaluate the model currently in memory
            self.evaluate(gaussian_model=gaussian_model, tag="after_train_model")
        elif choice == 'checkpoint':
            latest = self._find_latest_checkpoint()
            if latest:
                print(f"Evaluating using latest checkpoint: {latest}")
                self.evaluate(checkpoint_path=latest, tag="after_train_ckpt")
            else:
                print("No checkpoint found; falling back to in-memory model.")
                self.evaluate(gaussian_model=gaussian_model, tag="after_train_model")
        print("--- Post-Training Evaluation Finished ---")


    def evaluate(self, checkpoint_path=None, gaussian_model=None, tag="eval"):
        """
        Evaluate either:
        - an in-memory `gaussian_model` (use right after training), OR
        - a model loaded from `checkpoint_path`.

        If both are provided, `gaussian_model` takes precedence.
        """
        use_ckpt = (gaussian_model is None)

        if use_ckpt and checkpoint_path is None:
            raise ValueError("evaluate() needs either `gaussian_model` or `checkpoint_path`")

        if use_ckpt:
            print(f"Starting evaluation from checkpoint: {checkpoint_path}")
            gaussian_model = GaussianModel(logger=None, debug=False)
            loaded_iteration = self.load_checkpoint(gaussian_model, checkpoint_path)
            # If your renderer requires buffers set by training_setup(), keep this call:
            gaussian_model.training_setup(self.optim_params)
            iter_tag = loaded_iteration
        else:
            print("Starting evaluation using the in-memory model (no checkpoint load).")
            # Reasonable step for TensorBoard
            iter_tag = getattr(self.optim_params, "iterations", 0)

        # Create evaluation directory, allow unique tag names
        eval_dir_name = "evaluation" if tag is None else f"evaluation_{tag}"
        eval_plots_folder = os.path.join(self.run_folder, eval_dir_name)
        os.makedirs(eval_plots_folder, exist_ok=True)
        print(f"Saving evaluation plots to: {eval_plots_folder}")

        # --- Evaluation loop (unchanged except minor prints/vars) ---
        total_test_loss = 0.0
        total_test_samples = 0
        psnr_values, ssim_values, rssi_diff_values, mae_values, eval_render_times = [], [], [], [], []

        test_data_list = self.test_set.get_data()

        with torch.no_grad():
            progress_bar_eval = tqdm(enumerate(test_data_list), total=len(test_data_list), desc="Evaluating")
            for idx, test_data in progress_bar_eval:
                t0 = time.time()
                rendered_pkg = render(test_data, gaussian_model)
                t1 = time.time()
                current_render_time_ms = (t1 - t0) * 1000.0
                eval_render_times.append(t1 - t0)

                rendered_signal_real, _ = rendered_pkg['render']
                gt_signal_real = test_data.spectrum_real.cuda()

                if (idx % 2 == 0):
                    output_img = rendered_signal_real.detach().cpu().numpy()
                    gt_img = test_data.spectrum_real.cpu().numpy()
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1); plt.imshow(gt_img, cmap='gray'); plt.title('Ground Truth Spectrum'); plt.axis('off')
                    plt.subplot(1, 2, 2); plt.imshow(output_img, cmap='gray'); plt.title('Predicted Spectrum'); plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(eval_plots_folder, f"{idx}.png"))
                    plt.close()

                psnr_val = calculate_psnr(rendered_signal_real, gt_signal_real).item()
                rssi_diff_val = (calculate_rssi(rendered_signal_real) - calculate_rssi(gt_signal_real)).item()
                mae_val = calculate_mae(rendered_signal_real, gt_signal_real).item()
                ssim_val = ssim(rendered_signal_real, gt_signal_real).item()

                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                rssi_diff_values.append(rssi_diff_val)
                mae_values.append(mae_val)

                loss_l1_test = l1_loss(rendered_signal_real, gt_signal_real)
                loss_test = (1.0 - self.optim_params.lambda_ssim) * loss_l1_test + self.optim_params.lambda_ssim * (1.0 - ssim_val)
                total_test_loss += loss_test.item()
                total_test_samples += 1

                progress_bar_eval.set_postfix({
                    "SampleIdx": idx,
                    "PSNR": f"{psnr_val:.2f}",
                    "SSIM": f"{ssim_val:.3f}",
                    "MAE": f"{mae_val:.4f}",
                    "RenderTime(ms)": f"{current_render_time_ms:.2f}"
                })

        avg_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
        avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
        avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
        avg_rssi_diff = float(np.mean(rssi_diff_values)) if rssi_diff_values else 0.0
        avg_mae = float(np.mean(mae_values)) if mae_values else 0.0
        avg_render_time = (float(np.mean(eval_render_times)) * 1000.0) if eval_render_times else 0.0

        print("\n--- Evaluation Results ---")
        print(f"Eval source: {'checkpoint' if use_ckpt else 'in-memory model'}")
        print(f"Eval tag: {tag}")
        print(f"Average Test Loss: {avg_test_loss:.7f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average RSSI Difference: {avg_rssi_diff:.2f} dBm")
        print(f"Average MAE: {avg_mae:.7f}")
        print(f"Average Render Time: {avg_render_time:.2f} ms")
        print("--------------------------\n")

        self.writer.add_scalar('Loss/eval', avg_test_loss, iter_tag)
        # log_metric_distribution(self.writer, 'PSNR_eval', psnr_values, iter_tag, unit=' (dB)')
        log_metric_distribution(self.writer, 'SSIM_eval', ssim_values, iter_tag)
        # log_metric_distribution(self.writer, 'RSSI_Difference_eval', rssi_diff_values, iter_tag, unit=' (dBm)')
        # log_metric_distribution(self.writer, 'MAE_eval', mae_values, iter_tag)
        self.writer.add_scalar('Time/avg_render_eval_ms', avg_render_time, iter_tag)

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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Execution mode: train or eval') # Add 'eval' choice
    parser.add_argument('--dataset_type', type=str, default='mimo')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for densification and pruning')
    parser.add_argument('--num_test_plots', type=int, default=10, help='Number of test samples to plot after training (only used in train mode)') 
    parser.add_argument('--plot_initial_gaussians', action='store_true', default=False, help='Plot the initial distribution of Gaussians') # Add default=False
    parser.add_argument('--plot_final_gaussians', action='store_true', default=False, help='Plot the final distribution of Gaussians after training') # Add default=False
    parser.add_argument('--checkpoint_interval', type=int, default=200000, help='Interval for saving model checkpoints (train mode)') # Add checkpoint interval argument
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Subdirectory for saving/loading checkpoints') # Add checkpoint directory argument
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint file to load for evaluation (eval mode)') # Argument for loading checkpoint
    parser.add_argument('--eval_after_train', type=str, choices=['model','checkpoint','off'],default='model',help="After training, run evaluate() on: current 'model', latest 'checkpoint', or 'off'.")
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
                              eval_after_train=args.eval_after_train,
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

