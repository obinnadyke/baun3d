# devices.py: Device manager for memory management (c) itrustal.com
# multi-GPU setup, model info 

import torch
import torch.nn as nn
from thop import profile, clever_format
import gc
import os
import warnings
warnings.filterwarnings('ignore')
from loss_metric import get_loss_function


def check_all_devices(model, criterion, optimizer, device, verbose=True):
    """Check that all components are on correct device"""
    issues = []

    # Check model
    if verbose:
        print("\n" + "="*60)
        print("Checking Model...")
        print("="*60)

    # Unwrap DataParallel if needed
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    for name, param in base_model.named_parameters():
        if param.device.type == 'cpu':
            issues.append(f"Model param {name}: {param.device}")
            if verbose:
                print(f"  ✗ {name}: {param.device}")

    for name, buffer in base_model.named_buffers():
        # Skip fake buffers created by thop (FLOPs profiler)
        if 'total_ops' in name or 'total_params' in name:
            continue

        if buffer.device.type == 'cpu':
            issues.append(f"Model buffer {name}: {buffer.device}")
            if verbose:
                print(f"  ✗ {name}: {buffer.device}")

    if not issues:
        if verbose:
            print("  All model components on GPU")

    # Check criterion (loss function)
    if verbose:
        print("\n" + "="*60)
        print("Checking Criterion (Loss Function)...")
        print("="*60)

    if hasattr(criterion, 'parameters'):
        for name, param in criterion.named_parameters():
            if param.device.type == 'cpu':
                issues.append(f"Criterion param {name}: {param.device}")
                if verbose:
                    print(f"  ✗ {name}: {param.device}")

    if hasattr(criterion, 'buffers'):
        for name, buffer in criterion.named_buffers():
            # Skip fake buffers from thop
            if 'total_ops' in name or 'total_params' in name:
                continue

            if buffer.device.type == 'cpu':
                issues.append(f"Criterion buffer {name}: {buffer.device}")
                if verbose:
                    print(f"  ✗ {name}: {buffer.device}")

    # Check all submodules of criterion
    if hasattr(criterion, 'modules'):
        for module_name, module in criterion.named_modules():
            if module_name:  # Skip the root module
                for name, param in module.named_parameters(recurse=False):
                    if param.device.type == 'cpu':
                        issues.append(f"Criterion.{module_name} param {name}: {param.device}")
                        if verbose:
                            print(f"  ✗ {module_name}.{name}: {param.device}")

                for name, buffer in module.named_buffers(recurse=False):
                    # Skip fake buffers from thop
                    if 'total_ops' in name or 'total_params' in name:
                        continue

                    if buffer.device.type == 'cpu':
                        issues.append(f"Criterion.{module_name} buffer {name}: {buffer.device}")
                        if verbose:
                            print(f"  ✗ {module_name}.{name}: {buffer.device}")

    if len([i for i in issues if 'Criterion' in i]) == 0:
        if verbose:
            print("  All criterion components on GPU")

    # Summary
    if verbose:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)

    if issues:
        print(f"  Found {len(issues)} components on CPU")
        return False, issues
    else:
        print("  All components on GPU")
        return True, []


def force_all_to_device(model, criterion, device):
    """Safely move all components to device using standard PyTorch methods"""
    print(f"\nMoving all components to {device}...")

    # Move model using standard method
    model.to(device)

    # Move criterion using standard method
    criterion.to(device)

    # Verify with a forward pass
    dummy_input = torch.randn(1, 1, 32, 32, 32).to(device)
    try:
        with torch.no_grad():
            _ = model(dummy_input)
        print("  Verification forward pass successful")
    except Exception as e:
        print(f"  Warning: Verification failed: {e}")

    print("  All components safely moved to device")


class DeviceManager:
    """Manage GPU devices and memory"""

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.device = self.get_device()

        # Automatically enable DataParallel when batch_size >= 2 and multiple GPUs available
        num_gpus = torch.cuda.device_count()
        self.use_parallel = batch_size >= 2 and num_gpus > 1

        # Debug logging
        print(f"\n" + "="*60)
        print("DATAPARALLEL CONFIGURATION")
        print("="*60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Number of GPUs detected: {num_gpus}")
        if num_gpus > 0:
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"")
        print(f"Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Condition (batch_size >= 2): {batch_size >= 2}")
        print(f"  Condition (num_gpus > 1): {num_gpus > 1}")
        print(f"")
        print(f"DataParallel Status: {'ENABLED' if self.use_parallel else 'DISABLED'}")
        if not self.use_parallel and num_gpus > 1:
            print(f"  Reason: batch_size ({batch_size}) < 2")
        elif not self.use_parallel and num_gpus <= 1:
            print(f"  Reason: Only {num_gpus} GPU(s) available")
        print("="*60 + "\n")

    def get_device(self):
        """Get available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"Using primary GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            device = torch.device('cpu')
            print("Using CPU")
            return device

    def get_available_gpus(self):
        """Get list of available GPUs"""
        if not torch.cuda.is_available():
            return []

        n_gpus = torch.cuda.device_count()
        gpus = []

        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpus.append({
                'id': i,
                'name': gpu_name,
                'memory_gb': gpu_memory
            })

        return gpus

    def print_gpu_info(self):
        """Print GPU information"""
        gpus = self.get_available_gpus()

        if not gpus:
            print("No GPUs available")
            return

        print("\n" + "="*60)
        print("GPU Information:")
        print("="*60)
        for gpu in gpus:
            print(f"GPU {gpu['id']}: {gpu['name']}")
            print(f"  Total Memory: {gpu['memory_gb']:.2f} GB")
        print("="*60 + "\n")

    def setup_model(self, model):
        """Setup model with appropriate device configuration"""
        # Move model to device BEFORE DataParallel to prevent memory issues
        model = model.to(self.device)

        if self.use_parallel:
            device_ids = list(range(torch.cuda.device_count()))
            print(f"Setting up DataParallel with {len(device_ids)} GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
            print("DataParallel successfully applied")
            print("NOTE: Model forward calls will now distribute batches across all GPUs")
        else:
            print("\nUsing single GPU training")
            if torch.cuda.device_count() > 1:
                print(f"WARNING: {torch.cuda.device_count()} GPUs detected but DataParallel not enabled")
                print(f"         Increase batch_size to >= 2 to enable multi-GPU training")

        return model

    def cleanup_memory(self, aggressive=False):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                gc.collect()
                torch.cuda.empty_cache()


class ModelInfo:
    """
    Model information and statistics calculator.
    Provides parameter counts, model size, and FLOPs estimation.
    """

    def __init__(self, model, input_shape=None):
        """
        Initialize ModelInfo
        """
        self.model = model
        self.input_shape = input_shape
        # Unwrap DataParallel if needed
        self.base_model = model.module if isinstance(model, nn.DataParallel) else model
        self.is_parallel = isinstance(model, nn.DataParallel)

    def count_parameters(self):
        """
        Count total and trainable parameters in the model
        Returns tuple: (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def calculate_flops(self):
        """
        Calculate FLOPs and model size (Disabled to prevent thop from adding fake buffers to model)
        """
        return "Disabled", "Disabled"

    def get_model_size(self):
        """
        Get model size in megabytes (Model size in MB)
        """
        param_size = 0
        for param in self.base_model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in self.base_model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024**2)
        return size_mb

    def print_model_info(self, log_file=None):
        """
        Print comprehensive model information
        """
        total_params, trainable_params = self.count_parameters()
        model_size = self.get_model_size()

        info = []
        info.append("\n" + "="*60)
        info.append("Model Information:")
        info.append("="*60)
        info.append(f"Total Parameters: {total_params:,}")
        info.append(f"Trainable Parameters: {trainable_params:,}")
        info.append(f"Model Size: {model_size:.2f} MB")
        info.append(f"DataParallel Wrapped: {self.is_parallel}")

        try:
            flops, params = self.calculate_flops()
            info.append(f"FLOPs: {flops}")
            info.append(f"Params (calc): {params}")
        except:
            pass

        info.append("="*60 + "\n")

        msg = "\n".join(info)
        print(msg)

        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + "\n")


class CheckpointManager:
    """
    Manage model checkpoints for saving and loading model states
    """
    def __init__(self, output_dir):
        """
        Initialize checkpoint manager.
        output_dir (str): Base directory for saving checkpoints
        """
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_metrics, train_metrics, is_best_val=False, is_best_train=False):
        """
        Save model checkpoint with optimizer and scheduler states
        """
        # Add state_dict copy to prevent corruption during save
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict().copy()
        else:
            model_state = model.state_dict().copy()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics
        }

        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best validation model
        if is_best_val:
            best_val_path = os.path.join(self.checkpoint_dir, 'best_val_model.pth')
            torch.save(checkpoint, best_val_path)

        # Save best training model
        if is_best_train:
            best_train_path = os.path.join(self.checkpoint_dir, 'best_train_model.pth')
            torch.save(checkpoint, best_train_path)

        # Save epoch checkpoint every 50 epochs
        if epoch % 50 == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None, use_train_best=False):
        """
        Load model checkpoint
        """
        if checkpoint_path is None:
            if use_train_best:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_train_model.pth')
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_val_model.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle DataParallel
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


class EMA:
    """
    Exponential Moving Average for model weights - Maintains shadow parameters
    """

    def __init__(self, model, decay=0.999):
        """
        Initialize EMA
        Args:
            model (nn.Module): Model to track
            decay (float): EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = next(model.parameters()).device

        self.register()

    def register(self):
        """Register model parameters for EMA tracking"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        """Shadow parameters using exponential moving average"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone().to(self.device)

    def apply_shadow(self):
        """Apply shadow parameters to model (for validation/inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        """Restore original parameters (after validation/inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].to(param.device)
        self.backup = {}


def setup_training(config, model, output_dir):
    """
    Setup complete training environment with automatic DataParallel
    """
    # Import criterion after model is on device
    criterion = get_loss_function(config)

    # Device manager - automatically handles DataParallel based on batch_size
    device_manager = DeviceManager(batch_size=config.batch_size)
    device_manager.print_gpu_info()

    # Get target device
    device = device_manager.device

    # Move model to device using safe method
    print(f"\nMoving model to {device}...")
    model = model.to(device)
    criterion = criterion.to(device)

    # Verify with a forward pass
    dummy_input = torch.randn(1, config.in_channels, *config.patch_size).to(device)
    try:
        with torch.no_grad():
            _ = model(dummy_input)
        print("  Verification forward pass successful")
    except Exception as e:
        print(f"  Warning: Verification failed: {e}")

    # Verify all components are on correct device
    success, issues = check_all_devices(model, criterion, None, device, verbose=True)
    if not success:
        print(f"Warning: Some components on CPU: {issues}")
        force_all_to_device(model, criterion, device)

    # Apply DataParallel (model is already on device)
    if device_manager.use_parallel:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"\nApplying DataParallel with {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        print("DataParallel successfully applied")
        print("NOTE: Model forward calls will now distribute batches across all GPUs")
    else:
        print("\nUsing single GPU training")
        if torch.cuda.device_count() > 1:
            print(f"WARNING: {torch.cuda.device_count()} GPUs detected but DataParallel not enabled")
            print(f"         Increase batch_size to >= 2 to enable multi-GPU training")

    # Model info
    model_info = ModelInfo(model, input_shape=(1, config.in_channels, *config.patch_size))
    log_file = os.path.join(output_dir, 'model_info.txt')
    model_info.print_model_info(log_file)

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(output_dir)

    # EMA (optional but recommended)
    ema = None
    if config.use_ema:
        # Get base model (unwrap DataParallel if needed)
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        ema = EMA(base_model, decay=config.ema_decay)
        print("EMA initialized")

    return model, device_manager, checkpoint_manager, ema
