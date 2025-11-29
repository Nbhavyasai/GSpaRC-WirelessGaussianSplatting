#####Define optimizer and Loss function#####
###optimizer should support pruning and densification###

class OptimizationParams:
    def __init__(self, **kwargs):
        # Default values
        self.iterations = 400_000
        self.lambda_ssim = 0.2
        self.position_lr_init = 0.0016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 100_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.0055
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.emission_mlp_lr = 0.002
        self.percent_dense = 0.001
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 1000
        self.densify_until_iter = 50_000
        self.densify_grad_threshold = 0.0002
        self.cube_size = 1.8
        