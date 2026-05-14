#####Define optimizer and Loss function#####
###optimizer should support pruning and densification###

class OptimizationParams:
    def __init__(self, **kwargs):
        # Default values
        self.iterations = 200_000
        self.lambda1 = 0.2
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.00000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 200000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.0055
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.emission_mlp_lr = 0.002
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 300
        self.opacity_reset_interval = 4000
        self.densify_from_iter = 1000
        self.densify_until_iter = 120000
        self.densify_grad_threshold = 0.0002
        self.cube_size = 0.3

        # ── Confidence parameters (DUSt3R-style) ──
        # Learning rate for the confidence MLP
        self.confidence_lr = 0.001
        # Alpha controls the regularisation strength in L = C * loss - alpha * log(C)
        # Higher alpha → network is penalised more for being unconfident → pushes C up
        # Lower  alpha → network can more freely down-weight hard samples
        self.confidence_alpha = 0.3
        # Hidden layer size for the small rx_pos → confidence MLP
        self.confidence_hidden_size = 32

        # Override defaults with any values from the YAML config
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        