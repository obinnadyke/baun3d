# config.py - BAUN3D Data-Specific Parameter Settings (c) itrustal.com
import os


class BaseConfig:
    # Model Architecture
    in_channels = 1
    base_channels = 48
    num_stages = 5
    deep_supervision = True
    use_ema = True
    ema_decay = 0.995
    dropout_rate = 0.15

    # Tumor-specific
    tumor_label = 2  # tumor label
    organ_label = 1  # organ label

    # Optimization
    learning_rate = 5e-4
    base_lr = 5e-4
    weight_decay = 3e-5

    # Scheduler
    scheduler = 'cosine_warmup'
    warmup_epochs = 15
    use_reduce_on_plateau = True
    reduce_on_plateau_patience = 12
    reduce_on_plateau_factor = 0.5
    reduce_on_plateau_min_lr = 5e-6
    cosine_eta_min_factor = 0.05
    cosine_T_max = 100

    # Loss
    dice_weight = 0.25
    focal_tversky_weight = 0.45
    boundary_weight = 0.08

    # Tumor weighting
    tumor_loss_weight = 0.3
    tumor_ramp_epochs = 30
    tumor_gradient_multiplier = 1.5
    curriculum_epochs = 40
    tumor_loss_weight_start = 0.05
    tumor_loss_weight_end = 0.3

    # Boundary Loss Schedule
    boundary_warmup_start = 15
    boundary_warmup_epochs = 40
    boundary_max_weight = 0.5

    # Data Loading
    num_workers = 10
    pin_memory = True
    prefetch_factor = 3
    persistent_workers = True

    # Training Schedule
    gradient_accumulation_steps = 1
    gradient_clip = 1.0
    tumor_head_gradient_clip = 0.5
    mixed_precision = True
    val_interval = 10   # Frequency for viz monitoring

    # Augmentation
    aug_prob = 0.0
    elastic_deform = False
    elastic_alpha = (10, 50)
    elastic_sigma = (8, 10)
    gaussian_noise = False
    gaussian_noise_std = (0.0, 0.01)
    intensity_transforms = True
    brightness_range = (0.8, 1.2)
    contrast_range = (0.8, 1.2)
    geometric_transforms = True
    rotation_range = (-20, 20)
    scale_range = (0.9, 1.1)

    # MixUp
    use_mixup = False
    mixup_alpha = 0.2

    # Tumor Augmentation
    tumor_aug_prob = 0.5
    tumor_copy_paste = True

    # Caching
    cache_rate = 0.0

    # Split
    train_split = 0.7

    # Patch Sampling
    tumor_patch_ratio = 0.9
    organ_patch_ratio = 0.1
    random_patch_ratio = 0.0

    # Class Weights
    organ_weight = 0.10
    tumor_weight = 0.90

    # Validation
    val_eval_mode = 'sliding'
    val_window_size = (96, 128, 128)
    val_stride = (48, 64, 64)
    val_patch_until_epoch = 50
    fullval_every = 5
    val_max_samples = 15

    # Progressive Validation
    progressive_validation = True

    # Metric Smoothing
    use_metric_smoothing = True
    smoothing_alpha = 0.3

    # Logging
    log_gradients = False
    compute_val_hd95 = False
    fast_validation = True

#------------------------------------------------------------------------------

# Dataset-specific configs inheriting BaseConfig parameters
class LiTSConfig(BaseConfig):
    dataset_name = 'lits'
    num_classes = 3

    # Patch Sampling
    patch_size = (64, 96, 96)
    batch_size = 2
    epochs = 400

    # Class Weights
    organ_weight = 0.10
    tumor_weight = 0.90

    # Image Preprocessing
    window_level = 60
    window_width = 400
    clip_range = (-100, 400)
    target_spacing = (1.5, 1.5, 2.0)

    # Optimization
    learning_rate = 3e-4  # was 1e-4
    base_lr = 5e-4

    # Scheduling
    scheduler = 'cosine_warmup'
    use_reduce_on_plateau = False
    boundary_warmup_start = 20
    tumor_ramp_epochs = 30
    tumor_loss_weight_end = 0.3

#----------------------------LiTSConfig Ends-----------------------------------

class PancreasConfig(BaseConfig):
    dataset_name = 'pancreas'
    num_classes = 3
    organ_label = 1
    tumor_label = 2

    # Patch Sampling
    patch_size = (64, 96, 96)
    batch_size = 2
    epochs = 400

    # Tumor sampling
    tumor_patch_ratio = 0.70
    organ_patch_ratio = 0.30
    random_patch_ratio = 0.0

    # Strong tumor priority
    organ_weight = 0.30
    tumor_weight = 0.70

    # IMAGE PREPROCESSING
    window_level = 80
    window_width = 400
    clip_range = (-100, 350)
    target_spacing = (1.5, 1.5, 2.0)

    # OPTIMIZATION
    learning_rate = 5e-4     # was 8e-4 but (3e-4) recommended
    base_lr = 5e-4           # was 8e-4
    weight_decay = 1e-5      # Reduce regularization
    tumor_gradient_multiplier = 1.2

    # SCHEDULING
    scheduler = 'cosine_warmup'
    use_reduce_on_plateau = False
    boundary_warmup_start = 30
    boundary_warmup_epochs = 35
    tumor_ramp_epochs = 40
    tumor_loss_weight_end = 0.30    # Increased

    # GRADIENT CLIPPING
    gradient_clip = 0.5

    # PROGRESSIVE VAL
    progressive_validation = True
    val_patch_until_epoch = 50

    # CURRICULUM LEARNING
    use_curriculum = True
    curriculum_epochs = 40
    tumor_loss_weight_start = 0.15

#--------------------------PancreasConfig Ends---------------------------------

class ProstateConfig(BaseConfig):
    dataset_name = 'prostate'
    num_classes = 3
    organ_label = 1
    tumor_label = 2

    # Input Channels
    in_channels = 2

    # Patch Sampling
    patch_size = (32, 128, 128)
    batch_size = 2
    epochs = 350

    # Tumor sampling
    tumor_patch_ratio = 0.80
    organ_patch_ratio = 0.20
    random_patch_ratio = 0.0

    # Image Preprocessing (soft-tissue/pelvis)
    window_level = 70
    window_width = 350
    clip_range = (-150, 250)
    target_spacing = (1.0, 1.0, 2.0) # Higher XY resolution

    # OPTIMIZATION
    learning_rate = 5e-4
    base_lr = 5e-4
    tumor_gradient_multiplier = 1.5

    # Scheduling
    boundary_warmup_start = 27

#------------------------------ProstateConfig Ends-----------------------------


# Factory Functions
def get_config(dataset_name):
    """Factory function to retrieve dataset-specific configuration"""
    configs = {
        'lits': LiTSConfig,
        'pancreas': PancreasConfig,
        'prostate': ProstateConfig,
        'kits': BaseConfig,     # Add proper config if needed

    }

    if dataset_name.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")

    return configs[dataset_name.lower()]()

# Consistent naming
def tumor_loss_weight(config, current_epoch):
    """Calculate tumor loss weight with smooth curriculum"""
    if not getattr(config, 'use_curriculum', True):
        return getattr(config, 'tumor_loss_weight', 0.3)

    start_weight = getattr(config, 'tumor_loss_weight_start', 0.1)
    end_weight = getattr(config, 'tumor_loss_weight_end', 0.3)
    ramp_epochs = getattr(config, 'tumor_ramp_epochs', 30)

    if current_epoch >= ramp_epochs:
        return end_weight

    progress = current_epoch / ramp_epochs
    smooth_progress = progress ** 2  # Quadratic ramp
    weight = start_weight + (end_weight - start_weight) * smooth_progress

    return weight

def get_boundary_weight_schedule(config, current_epoch):
    """Calculate boundary loss weight with delayed start"""
    if current_epoch < config.boundary_warmup_start:
        return 0.0

    progress = (current_epoch - config.boundary_warmup_start) / max(1, config.boundary_warmup_epochs)
    smooth_progress = progress ** 2
    return config.boundary_max_weight * max(0.0, min(0.5, smooth_progress))
