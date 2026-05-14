# train.py - Main training and validation Script | (c) itrustal.com [2026]

import os
import argparse
import warnings
import faulthandler  # segfault debugging
warnings.filterwarnings('ignore')

# Set environment variables BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OMP_NUM_THREADS'] = '10'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import sys
#import signal
import traceback
import gc
import time
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import label as label_components

#from scipy.ndimage import gaussian_filter
#from monai.inferers import sliding_window_inference
#from monai.transforms import KeepLargestConnectedComponent

from config import get_config, get_boundary_weight_schedule, tumor_loss_weight
from dataloader import get_dataloader
from devices import setup_training
from loss_metric import get_loss_function, get_metric_calculator
from model import build_model
from plotter import (
    analyze_dataset,
    plot_training_curves,
    visualize_segmentation_comparison_multiview,
    generate_post_training_analysis,
)

ORGAN_NAMES = {
    'lits': 'Liver',
    'pancreas': 'Pancreas',
    'kits': 'Kidney',
    'prostate': 'Prostate'
}


# Prediction helper function --------------------
def clean_prediction(pred, organ_label, tumor_label):
    """Remove false positive via connected component analysis"""
    cleaned = np.zeros_like(pred)

    organ_mask = (pred == organ_label)
    if organ_mask.sum() > 0:
        labeled, num = label_components(organ_mask)
        if num > 0:
            largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            cleaned[labeled == largest] = organ_label

    tumor_mask = (pred == tumor_label)
    if tumor_mask.sum() > 0:
        labeled, num = label_components(tumor_mask)
        for i in range(1, num + 1):
            comp = (labeled == i)
            if comp.sum() >= 10:
                cleaned[comp] = tumor_label

    return cleaned
#---------------------------------------------------


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.005, consider_tumor=True):
        self.patience = patience
        self.min_delta = min_delta
        self.consider_tumor = consider_tumor
        self.counter = 0
        self.best = -np.inf
        self.best_tumor = -np.inf
        self.best_epoch = 0

    def __call__(self, avg_dice, tumor_dice=None, epoch=None):
        improved = avg_dice > self.best + self.min_delta

        if self.consider_tumor and tumor_dice is not None:
            tumor_improved = tumor_dice > self.best_tumor + self.min_delta
            improved = improved and tumor_improved  # was 'or'

            if tumor_improved:
                self.best_tumor = tumor_dice

        if improved:
            self.best = avg_dice
            self.best_epoch = epoch or 0
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    # Training Coordinator Function
    def __init__(self, config, args):
        # Enable fault handler for segfault debugging
        faulthandler.enable()

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()

        self.config = config
        self.args = args
        self.organ_name = ORGAN_NAMES.get(config.dataset_name.lower(), 'Organ')

        self.output_dir = os.path.join(args.output_dir, config.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, 'training.log')
        self.viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

        print("Building model...")
        model = build_model(config)

        if torch.cuda.device_count() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print(f"Using {torch.cuda.device_count()} GPUs with SyncBatchNorm")

        self.model, self.device_manager, self.checkpoint_manager, self.ema = setup_training(
            config, model, self.output_dir
        )
        self.device = self.device_manager.device

        self.criterion = get_loss_function(config)
        self.metric_calc = get_metric_calculator(config)

        # Setup optimizer with parameter groups
        base_params = []
        tumor_head_params = []

        for name, param in self.model.named_parameters():
            if any(x in name for x in ['head', 'aux', 'tumor', 'decoder', 'ref']):
                tumor_head_params.append(param)
            else:
                base_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': config.learning_rate, 'weight_decay': config.weight_decay},
            {'params': tumor_head_params, 'lr': config.learning_rate * 1.5, 'weight_decay': config.weight_decay * 0.5}
        ], betas=(0.9, 0.999), eps=1e-8)

        self.scaler = GradScaler() if config.mixed_precision else None

        self.current_epoch = 0
        self.best_val_dice = -1.0
        self.best_train_dice = -1.0

        self.history = {
            'train_loss': [], 'train_organ_dice': [], 'train_tumor_dice': [], 'train_avg_dice': [],
            'val_loss': [], 'val_organ_dice': [], 'val_tumor_dice': [], 'val_avg_dice': [],
            'learning_rate': [], 'boundary_strength': [], 'curriculum_weight': []
        }

        # 3: Correct scheduler variable references
        self._setup_schedulers()
        self.early_stopping = EarlyStopping(patience=30, min_delta=0.005, consider_tumor=True)

        # Explicitly initialize current_scheduler attribute
        self.current_scheduler = self.primary_scheduler

        self.val_loader = None
        self.boundary_warmup_epochs = config.boundary_warmup_epochs
        self.use_boundary_refinement = False
        self.current_boundary_strength = 0.0
        self.smoothed_val_dice = None

    def _setup_schedulers(self):
        """
        Setup learning rate schedulers with fallback logic
        """
        warmup_epochs = getattr(self.config, 'warmup_epochs', 10)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.05, total_iters=warmup_epochs
        )

        # Primary scheduler (ReduceLROnPlateau) - Stationary LR until plateau
        self.primary_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.config.reduce_on_plateau_factor,
            patience=self.config.reduce_on_plateau_patience,
            min_lr=self.config.reduce_on_plateau_min_lr,
            verbose=True
        )

        # Fallback scheduler (CosineAnnealingLR) - Smooth Decay
        self.fallback_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.cosine_T_max,
            eta_min=self.config.learning_rate * self.config.cosine_eta_min_factor
        )

        # --- Use settings in config Script ---
        if getattr(self.config, 'use_reduce_on_plateau', True):
            self.scheduler_stage = 'primary'
            self.current_scheduler = self.primary_scheduler
            print("Scheduler: Active Strategy -> ReduceLROnPlateau")
        else:
            # Ensure use of Cosine Decay
            self.scheduler_stage = 'fallback'
            self.current_scheduler = self.fallback_scheduler
            print("Scheduler: Active Strategy -> CosineAnnealingLR")

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def model_forward(self, images, use_refinement=False, boundary_strength=1.0):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.use_refinement = use_refinement
            self.model.module.boundary_strength = boundary_strength
        else:
            self.model.use_refinement = use_refinement
            self.model.boundary_strength = boundary_strength
        return self.model(images)

    def _update_config_weights(self, epoch):
        current_tumor_weight = tumor_loss_weight(self.config, epoch)
        self.config.tumor_weight = current_tumor_weight
        self.config.organ_weight = 1.0 - current_tumor_weight
        self.criterion.config.tumor_weight = current_tumor_weight
        self.criterion.config.organ_weight = self.config.organ_weight

    def compute_curriculum_loss(self, outputs, labels, contours, epoch):
        if isinstance(outputs, (list, tuple)):
            pred_logits = outputs[0]
            aux_outputs = outputs[1] if len(outputs) > 1 else []
        else:
            pred_logits = outputs
            aux_outputs = []

        pred_logits, contours = self._align_spatial(pred_logits, labels, contours)

        boundary_weight = get_boundary_weight_schedule(self.config, epoch)
        use_boundary = epoch > self.config.boundary_warmup_start and boundary_weight > 0

        main_loss = self.criterion(pred_logits, labels, contours, use_boundary=use_boundary, epoch=epoch)

        if aux_outputs and self.model.training:
            aux_loss = 0.0
            for aux_out in aux_outputs:
                aux_out_aligned = F.interpolate(aux_out, size=labels.shape[1:], mode='trilinear', align_corners=False)

                # Handle contours alignment - keep as 5D [B, C, D, H, W]
                if contours.ndim == 5:
                    contours_aligned = F.interpolate(contours.float(), size=labels.shape[1:], mode='trilinear', align_corners=False)
                else:
                    contours_aligned = F.interpolate(contours.unsqueeze(1).float(), size=labels.shape[1:], mode='trilinear', align_corners=False)

                aux_loss += self.criterion(aux_out_aligned, labels, contours_aligned, use_boundary=False, epoch=epoch)

            aux_loss *= 0.3
            return main_loss + aux_loss

        return main_loss

    def _align_spatial(self, pred, target, contours):
        if pred.shape[2:] != target.shape[1:]:
            pred = F.interpolate(pred, size=target.shape[1:], mode='trilinear', align_corners=False)

        # Contour Handler (ensure channel dimension and correct spatial size)
        if contours.ndim == 5:
            if contours.shape[2:] != target.shape[1:]:
                contours = F.interpolate(contours.float(), size=target.shape[1:], mode='trilinear', align_corners=False)
        elif contours.ndim == 4:
            if contours.shape[1:] != target.shape[1:]:
                contours = F.interpolate(contours.unsqueeze(1).float(), size=target.shape[1:], mode='trilinear', align_corners=False)
            else:
                contours = contours.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected contours shape: {contours.shape}")

        return pred, contours

    def train_epoch(self, data_loader, epoch, compute_metrics=False):
        self.model.train()
        self._update_config_weights(epoch)

        running_loss = 0.0
        all_organ_dice = []
        all_tumor_dice = []

        pbar = tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            contours = batch['contour'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # --- ROBUST TRAINING LOOP WITH GUARDRAILS ---
            if self.scaler:
                # 1) Forward under autocast
                with autocast():
                    outputs = self.model_forward(images, self.use_boundary_refinement, self.current_boundary_strength)

                # 2) Compute loss (Float32 is enforced inside criterion via loss_metric.py)
                loss = self.compute_curriculum_loss(outputs, labels, contours, epoch)

                # 3) Guard: Check for Non-Finite Loss BEFORE Backward
                if not torch.isfinite(loss):
                    print(f"[WARNING] Epoch {epoch} - Non-finite LOSS. Skipping batch {batch_idx}. loss={loss.item()}")
                    #self.scaler.update() # Back off scale
                    continue

                # 4) Backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # 5) Clip Grads
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                # 6) Guard: Check for Non-Finite Grads
                bad_param = None
                for name, p in self.model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_param = name
                        break

                if bad_param is not None:
                    print(f"[WARNING] Epoch {epoch} - Non-finite grads in '{bad_param}'. Skipping step for batch {batch_idx}.")
                    self.scaler.update() # Skip step, update scaler to reduce scale
                    continue

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard Precision Path
                outputs = self.model_forward(images, self.use_boundary_refinement, self.current_boundary_strength)
                loss = self.compute_curriculum_loss(outputs, labels, contours, epoch)

                if not torch.isfinite(loss):
                    print(f"[WARNING] Epoch {epoch} - Non-finite LOSS (FP32). Skipping batch {batch_idx}. loss={loss.item()}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                bad_param = None
                for name, p in self.model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_param = name
                        break

                if bad_param is not None:
                    print(f"[WARNING] Epoch {epoch} - Non-finite grads (FP32) in '{bad_param}'. Skipping step.")
                    continue

                self.optimizer.step()
#--------------------------------------------------------------------------
            if self.ema is not None:
                self.ema.update()

            running_loss += loss.item()

            if compute_metrics and batch_idx % 5 == 0:
                with torch.no_grad():
                    pred_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    pred = torch.argmax(pred_logits, dim=1).cpu().numpy()
                    target = labels.cpu().numpy()

                    for i in range(pred.shape[0]):
                        metrics = self.metric_calc.compute_all_metrics(
                            pred[i], target[i],
                            self.config.organ_label, self.config.tumor_label,
                            fast_mode=True
                        )
                        all_organ_dice.append(metrics['organ_dice'])
                        all_tumor_dice.append(metrics['tumor_dice'])

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.get_current_lr():.6f}"
            })

        avg_loss = running_loss / len(data_loader)

        if compute_metrics and len(all_organ_dice) > 0:
            organ_dice = np.mean(all_organ_dice)
            tumor_dice = np.mean(all_tumor_dice)
            avg_dice = (organ_dice + tumor_dice) / 2
        else:
            organ_dice = tumor_dice = avg_dice = None

        return avg_loss, organ_dice, tumor_dice, avg_dice

#-------------------------Visualizer starts_here-----------------------------

    def _get_gaussian(self, patch_size, sigma_scale=1. / 8):
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = scipy.ndimage.gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

        # Normalize to max 1
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # Clip minimum to avoid zero division in edge cases
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])

        return torch.from_numpy(gaussian_importance_map)


    def _predict_whole_volume(self, image_tensor):
        """
        Robust Sliding Window Inference (Handles small volumes via padding)
        """
        self.model.eval()

        # Ensure input is [1, C, D, H, W]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(0)

        b, c, D_orig, H_orig, W_orig = image_tensor.shape
        patch_size = self.config.patch_size
        num_classes = self.config.num_classes

        # 1. PADDING
        pd, ph, pw = patch_size
        pad_d = max(0, pd - D_orig)
        pad_h = max(0, ph - H_orig)
        pad_w = max(0, pw - W_orig)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Pad format: (left, right, top, bottom, front, back) -> W, H, D
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='replicate')

        _, _, D, H, W = image_tensor.shape

        # 2. CPU AGGREGATION
        final_logit = torch.zeros((b, num_classes, D, H, W), device='cpu', dtype=torch.float32)
        count_map = torch.zeros((D, H, W), device='cpu', dtype=torch.float32)

        # 3. STRIDE & STEPS
        stride = [max(1, s // 2) for s in patch_size]
        sd, sh, sw = stride

        z_steps = list(range(0, D - pd + 1, sd))
        y_steps = list(range(0, H - ph + 1, sh))
        x_steps = list(range(0, W - pw + 1, sw))

        # Add end steps (safe because D >= pd due to padding)
        if z_steps[-1] != D - pd: z_steps.append(D - pd)
        if y_steps[-1] != H - ph: y_steps.append(H - ph)
        if x_steps[-1] != W - pw: x_steps.append(W - pw)

        # 4. INFERENCE LOOP
        with torch.no_grad():
            for z in z_steps:
                for y in y_steps:
                    for x in x_steps:
                        z_end, y_end, x_end = z + pd, y + ph, x + pw

                        patch = image_tensor[:, :, z:z_end, y:y_end, x:x_end].to(self.device)

                        pred = self.model_forward(patch, use_refinement=False, boundary_strength=0.0)
                        if isinstance(pred, tuple): pred = pred[0]

                        # Softmax for smoother blending
                        pred = torch.softmax(pred, dim=1).cpu()

                        final_logit[:, :, z:z_end, y:y_end, x:x_end] += pred
                        count_map[z:z_end, y:y_end, x:x_end] += 1.0

        # 5. FINALIZE
        final_logit /= (count_map.unsqueeze(0).unsqueeze(0) + 1e-8)

        # Crop back to original size
        final_logit = final_logit[:, :, :D_orig, :H_orig, :W_orig]

        return final_logit


    def generate_visualization(self, epoch):
        print(f"\n[Visualization] Generating 3x3 grid for epoch {epoch}...")
        self.model.eval()

        # Reset refinement flags
        original_use_refinement = self.use_boundary_refinement
        self.use_boundary_refinement = False

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        try:
            ds = self.val_loader.dataset

            # --- FIND A CASE WITH TUMOR ---
            # Scan up to 10 random files to find one with a tumor
            found_good_case = False
            indices = np.random.permutation(len(ds.image_files))[:10]

            for sample_idx in indices:
                img_path = ds.image_files[sample_idx]
                lbl_path = ds.label_files[sample_idx]

                # Quick check using Nibabel before full load
                raw_img, raw_lbl, spacing = ds._load_case(img_path, lbl_path)

                if (raw_lbl == self.config.tumor_label).sum() > 500: # Threshold for "good" tumor
                    found_good_case = True
                    print(f"[Visualization] Selected case: {os.path.basename(img_path)}")
                    break

            if not found_good_case:
                print("[Visualization] Warning: No substantial tumor found in random search, using last loaded.")

            # Preprocess
            proc_img, proc_lbl = ds.preprocessor.preprocess(raw_img, raw_lbl, spacing)
            # Add channel dim for predict_whole_volume [D,H,W] -> [1,D,H,W] -> [1,1,D,H,W]
            images = torch.from_numpy(proc_img).unsqueeze(0).unsqueeze(0) if proc_img.ndim==3 else torch.from_numpy(proc_img).unsqueeze(0)

            # Use Robust CPU-based Sliding Window
            pred_logits = self._predict_whole_volume(images)

            pred = torch.argmax(pred_logits, dim=1)[0].numpy()

            pred = clean_prediction(pred, self.config.organ_label, self.config.tumor_label) # mop-up artifacts

            target = proc_lbl
            img_vol = proc_img[0] if proc_img.ndim==4 else proc_img

            # Calculate Metrics
            metrics = self.metric_calc.compute_all_metrics(
                pred, target,
                self.config.organ_label, self.config.tumor_label,
                fast_mode=True
            )

            # Plot
            viz_path = os.path.join(self.viz_dir, f'seg_epoch_{epoch:03d}.png')
            import matplotlib
            matplotlib.use('Agg', force=True)

            success = visualize_segmentation_comparison_multiview(
                img_vol, target, pred,
                self.config.organ_label, self.config.tumor_label,
                metrics=metrics, out_path=viz_path
            )

            if success: print(f"[Visualization] Saved to {viz_path}")

        except Exception as e:
            print(f"[Visualization Error] {e}")
            traceback.print_exc()

        finally:
            self.use_boundary_refinement = original_use_refinement
            if hasattr(self.model, 'module'):
                self.model.module.use_refinement = original_use_refinement
            else:
                self.model.use_refinement = original_use_refinement
            plt.close('all')
            gc.collect()
#-------------------------Viz---Ends_Here-------------------


    def validate(self, data_loader, epoch):
        """Validation """
        self.model.eval()

        running_loss = 0.0
        all_organ_dice = []
        all_tumor_dice = []

        val_sample_limit = min(len(data_loader), 20)

        with torch.no_grad():
            # Create manual tqdm progress bar with correct total
            pbar = tqdm(total=val_sample_limit, desc=f"Val Epoch {epoch}", leave=True)

            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= val_sample_limit:
                    break

                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                contours = batch['contour'].to(self.device, non_blocking=True)

                outputs = self.model_forward(images, False, 0.0)
                pred_logits = outputs[0] if isinstance(outputs, tuple) else outputs

                pred_logits, contours = self._align_spatial(pred_logits, labels, contours)

                loss = self.criterion(pred_logits, labels, contours, use_boundary=False, epoch=epoch)
                running_loss += loss.item()

                pred = torch.argmax(pred_logits, dim=1).cpu().numpy()
                target = labels.cpu().numpy()

                for i in range(pred.shape[0]):
                    metrics = self.metric_calc.compute_all_metrics(
                        pred[i], target[i],
                        self.config.organ_label, self.config.tumor_label,
                        fast_mode=True
                    )
                    all_organ_dice.append(metrics['organ_dice'])
                    all_tumor_dice.append(metrics['tumor_dice'])

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'org_dice': f"{np.mean(all_organ_dice):.3f}" if all_organ_dice else 0.0,
                    'tum_dice': f"{np.mean(all_tumor_dice):.3f}" if all_tumor_dice else 0.0
                })

                # Manually update the progress bar
                pbar.update(1)

            # close progress bar
            pbar.close()

        avg_loss = running_loss / val_sample_limit
        organ_dice = np.mean(all_organ_dice) if all_organ_dice else 0.0
        tumor_dice = np.mean(all_tumor_dice) if all_tumor_dice else 0.0
        avg_dice = (organ_dice + tumor_dice) / 2

        return avg_loss, organ_dice, tumor_dice, avg_dice


    def train(self, train_loader, val_loader, total_epochs, start_epoch=0):
        self.val_loader = val_loader
        self.current_epoch = start_epoch

        for epoch in range(start_epoch + 1, total_epochs + 1):
            self.current_epoch = epoch
            self._update_boundary_strength(epoch)

            # 1. Training Step
            compute_train_metrics = (epoch % 5 == 0) or (epoch <= 5)
            train_loss, tr_o, tr_t, tr_a = self.train_epoch(train_loader, epoch, compute_metrics=compute_train_metrics)

            # 2. Scheduler Step (Warmup)
            if epoch <= getattr(self.config, 'warmup_epochs', 10):
                self.warmup_scheduler.step()

            # 3. Validation Step (with visualization BEFORE validation)
            should_validate = (epoch % self.config.val_interval == 0) or (epoch == total_epochs)
            val_loss, val_o, val_t, val_a = None, None, None, None

            if should_validate:
                # Generate visualization FIRST, then validate
                try:
                    self.generate_visualization(epoch)
                except Exception as viz_error:
                    print(f"[Warning] Visualization failed but continuing: {viz_error}")

                # Now run validation
                val_loss, val_o, val_t, val_a = self.validate(val_loader, epoch)

                # 4. Scheduler Step (Post-Warmup)
                if self.scheduler_stage == 'primary' and epoch > getattr(self.config, 'warmup_epochs', 10):
                    scheduler_metric = (0.4 * val_loss) + (0.6 * (1 - val_a))
                    self.primary_scheduler.step(scheduler_metric)
                elif self.scheduler_stage == 'fallback':
                    self.fallback_scheduler.step()

                self.log(f"\n{'='*40}")
                self.log(f"Epoch {epoch}/{self.config.epochs} Summary")
                self.log(f"{'='*40}")

                self._log_epoch_results(epoch, train_loss, tr_o, tr_t, tr_a, val_loss, val_o, val_t, val_a)

            # 5. Save Checkpoint after validation
            if should_validate and val_o is not None:
                self._save_checkpoint(epoch, val_o, val_t, val_a, tr_o, tr_t, tr_a)

                if val_a is not None:
                    if self.early_stopping(val_a, val_t, epoch):
                        self.log(f"\nEarly stopping triggered at epoch {epoch}")
                        break

            # Periodic cleanup every 10 epochs
            if epoch % 10 == 0:
                self.cleanup_memory()

            # Re-enable boundary refinement after validation
            if epoch >= self.config.boundary_warmup_start:
                self._update_boundary_strength(epoch)

        self._finalize_training()

    def _update_boundary_strength(self, epoch):
        """
        Boundary strength activation function
        """
        #warmup_start = getattr(self.config, 'boundary_warmup_start', 26)
        warmup_start = getattr(self.config, 'boundary_warmup_start', None)
        if warmup_start is None:
            raise ValueError("boundary_warmup_start must be defined in config.py")

        if epoch < warmup_start:
            self.current_boundary_strength = 0.0
            self.use_boundary_refinement = False

            # Ensure model's internal flag is disabled
            if hasattr(self.model, 'module'):
                self.model.module.use_refinement = False
            else:
                self.model.use_refinement = False
        else:
            progress = (epoch - warmup_start) / max(1, self.boundary_warmup_epochs)
            self.current_boundary_strength = min(0.3, progress ** 2 * 0.3)  # Reduced max strength
            self.use_boundary_refinement = True

            # Now enable model's internal flag
            if hasattr(self.model, 'module'):
                self.model.module.use_refinement = True
            else:
                self.model.use_refinement = True

    def _log_epoch_results(self, epoch, train_loss, tr_o, tr_t, tr_a, val_loss, val_o, val_t, val_a):
        self.log(f"\nEpoch {epoch}/{self.config.epochs}")
        self.log(f"  LR: {self.get_current_lr():.6f} | Boundary: {self.current_boundary_strength:.3f}")
        self.log(f"  Tumor Weight: {self.config.tumor_weight:.4f}")
        self.log(f"  Train Loss: {train_loss:.4f}")

        if tr_o is not None:
            self.log(f"  Train {self.organ_name} Dice: {tr_o:.4f}")
            self.log(f"  Train Tumor Dice: {tr_t:.4f}")
            self.log(f"  Train Avg Dice: {tr_a:.4f}")

        if val_loss is not None:
            self.log(f"  Val Loss: {val_loss:.4f}")
            self.log(f"  Val {self.organ_name} Dice: {val_o:.4f}")
            self.log(f"  Val Tumor Dice: {val_t:.4f}")
            self.log(f"  Val Avg Dice: {val_a:.4f}")

        self.history['train_loss'].append(train_loss)
        if tr_o is not None:
            self.history['train_organ_dice'].append(tr_o)
            self.history['train_tumor_dice'].append(tr_t)
            self.history['train_avg_dice'].append(tr_a)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            self.history['val_organ_dice'].append(val_o)
            self.history['val_tumor_dice'].append(val_t)
            self.history['val_avg_dice'].append(val_a)

        self.history['learning_rate'].append(self.get_current_lr())
        self.history['boundary_strength'].append(self.current_boundary_strength)
        self.history['curriculum_weight'].append(tumor_loss_weight(self.config, epoch))

    def _save_checkpoint(self, epoch, val_o, val_t, val_a, tr_o, tr_t, tr_a):
        """Save checkpoint with CORRECT scheduler reference"""
        is_best_val = val_a > self.best_val_dice + 0.001
        is_best_train = tr_a is not None and tr_a > self.best_train_dice + 0.001

        if is_best_val:
            self.best_val_dice = val_a
            self.log(f"  *** New best VAL: {val_a:.4f} (Tumor: {val_t:.4f}) ***")

        if is_best_train and tr_a is not None:
            self.best_train_dice = tr_a
            self.log(f"  *** New best TRAIN: {tr_a:.4f} (Tumor: {tr_t:.4f}) ***")

        should_save = is_best_val or is_best_train

        if should_save:
            val_metrics = {'organ_dice': val_o, 'tumor_dice': val_t, 'avg_dice': val_a}
            train_metrics = {'organ_dice': tr_o or 0.0, 'tumor_dice': tr_t or 0.0, 'avg_dice': tr_a or 0.0}

            # Use self.current_scheduler which is guaranteed to exist
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.current_scheduler,
                epoch, val_metrics, train_metrics, is_best_val, is_best_train
            )

    def _finalize_training(self):
        print("\nGenerating final plots...")
        plot_training_curves(self.history, self.output_dir, organ_name=self.organ_name)

        # --- Compute Final METRICS ---
        try:
            print("\n--- Computing All Comprehensive Metrics ---")

            # Load best validation model (optional, but recommended for final metrics)
            best_checkpoint = self.checkpoint_manager.load_checkpoint(
                self.model, use_train_best=False
            )
            if best_checkpoint:
                print(f"Loaded best model from epoch {best_checkpoint.get('epoch', 'N/A')}.")

            # Run full validation pass (or a large subset) using slow metrics
            final_metrics, predictions = self._compute_final_metrics(self.val_loader, self.config.compute_val_hd95)

            self._log_final_metrics(final_metrics)

            # Pass predictions to post-training analysis
            generate_post_training_analysis(self.config, self.output_dir, predictions=predictions)

        except Exception as e:
            print(f"[FINAL METRICS ERROR] Computing final metrics failed: {e}")
            traceback.print_exc()


        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Dice: {self.best_val_dice:.4f}")
        print(f"Best Train Dice: {self.best_train_dice:.4f}")
        print(f"Results saved to: {self.output_dir}")


    def _compute_final_metrics(self, data_loader, compute_hd95=False):
        """Run full evaluation and compute all slow metrics."""
        self.model.eval()

        all_metrics = []
        all_predictions = []    # Collect predictions
        all_targets = []        # Collect GT
        val_sample_limit = len(data_loader) # Use all samples for final evaluation

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Final Evaluation", leave=True, total=val_sample_limit)

            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= val_sample_limit:
                    break

                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                # Use the full prediction pipeline
                images_cpu = images.cpu().unsqueeze(0) if images.ndim == 4 else images.cpu()
                pred_logits = self._predict_whole_volume(images_cpu)

                # Prediction and Target for metric calculation
                pred = torch.argmax(pred_logits, dim=1)[0].numpy()
                target = labels.cpu().numpy()[0]

                # Collect predictions for confusion matrix (sample subset)
                if batch_idx < 20:  # Sample first 20 cases
                    all_predictions.append(pred)
                    all_targets.append(target)

                # Determine spacing for HD95/ASD calculation
                if 'target_spacing_zyx' in batch:
                    # Device agnostic batch (sx,sy,sz) extraction from dataloader
                    spacing = tuple(batch['target_spacing_zyx'][0].detach().cpu().numpy())
                else:
                    # Fallback to config if batch info missing
                    spacing = tuple(getattr(self.config, 'target_spacing', (1.0, 1.0, 2.0))[::-1])

                # Ensure spacing is a tuple of floats for the metric calculator
                spacing = tuple(float(s) for s in spacing)

                # Compute ALL metrics (fast_mode=False)
                metrics = self.metric_calc.compute_all_metrics(
                    pred, target,
                    self.config.organ_label, self.config.tumor_label,
                    spacing=spacing,
                    fast_mode=False
                )
                all_metrics.append(metrics)

        # Average all metrics
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

        # Prep prediction dict
        predictions = {
            'y_true': np.concatenate(all_targets) if all_targets else None,
            'y_pred': np.concatenate(all_predictions) if all_predictions else None
        }
        return avg_metrics, predictions


    def _log_final_metrics(self, metrics):
        """
        Formats and writes all final metrics to the training log
        """
        self.log("\n" + "="*60)
        self.log("COMPREHENSIVE FINAL METRICS (AVG)")
        self.log("="*60)

        # 1. Dice & Volume Overlap
        self.log("--- I. OVERLAP AND VOLUME METRICS ---")
        self.log(f"| {self.organ_name} Dice: {metrics['organ_dice']:.4f}")
        self.log(f"| Tumor Dice:    {metrics['tumor_dice']:.4f}")
        self.log(f"| Average Dice:  {metrics['avg_dice']:.4f}")
        self.log(f"| {self.organ_name} IoU:  {metrics['organ_iou']:.4f}")
        self.log(f"| Tumor IoU:     {metrics['tumor_iou']:.4f}")

        # 2. Sensitivity & Precision
        self.log("\n--- II. RECALL AND PRECISION METRICS ---")
        self.log(f"| {self.organ_name} Sensitivity (Recall): {metrics['organ_sensitivity']:.4f}")
        self.log(f"| Tumor Sensitivity (Recall):    {metrics['tumor_sensitivity']:.4f}")
        self.log(f"| {self.organ_name} Precision:            {metrics['organ_precision']:.4f}")
        self.log(f"| Tumor Precision:               {metrics['tumor_precision']:.4f}")

        # 3. Surface Distance
        self.log("\n--- III. SURFACE DISTANCE METRICS (MM) ---")
        self.log(f"| {self.organ_name} HD95: {metrics['organ_hd95']:.2f}")
        self.log(f"| Tumor HD95:    {metrics['tumor_hd95']:.2f}")
        self.log(f"| {self.organ_name} ASD:  {metrics['organ_asd']:.2f}")
        self.log(f"| Tumor ASD:     {metrics['tumor_asd']:.2f}")
        self.log("="*60 + "\n")

    def cleanup_memory(self):
        """Memory cleanup with matplotlib support"""
        print("[Cleanup] Running memory cleanup...")

        # Close all matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        print("[Cleanup] Complete")


def main():
    parser = argparse.ArgumentParser(description='BAUN3D Medical Segmentation')
    parser.add_argument('--dataset', type=str, required=True, choices=['lits', 'pancreas', 'kits', 'prostate'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--skip_analysis', action='store_true')

    args = parser.parse_args()

    # LOAD CONFIG
    config = get_config(args.dataset)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs

    # DATASET ANALYSIS
    if not args.skip_analysis:
        print("\nAnalyzing dataset...")
        analyze_dataset(config, args.data_dir, args.output_dir)

    # CREATE DATA LOADERS
    print("Creating Data Loaders...")

    train_loader = get_dataloader(config, args.data_dir, mode='train', batch_size=config.batch_size)
    val_config = get_config(args.dataset)
    val_config.num_workers = max(1, config.num_workers // 2)
    val_loader = get_dataloader(val_config, args.data_dir, mode='val', batch_size=1)

    # INITIATE TRAINING
    print("Initializing Trainer")

    trainer = Trainer(config, args)

    start_epoch = 0
    if args.resume:
        print(f"[RESUME] Loading: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')

        model_to_load = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and trainer.current_scheduler:
            trainer.current_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        trainer.best_val_dice = checkpoint.get('best_val_dice', -1.0)
        print(f"Resumed from epoch {start_epoch}")

    print(f"\n" + "="*40)
    print(f"Starting training: {config.dataset_name.upper()}")
    print(f"Epochs: {config.epochs} | Batch Size: {config.batch_size}")
    print(f"GPUs: {torch.cuda.device_count()}")
    print("="*40 + "\n")

    try:
        trainer.train(train_loader, val_loader, config.epochs, start_epoch)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving emergency checkpoint...")
        checkpoint = {
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.module.state_dict() if isinstance(trainer.model, nn.DataParallel) else trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.current_scheduler.state_dict(),
            'best_val_dice': trainer.best_val_dice
        }
        torch.save(checkpoint, os.path.join(trainer.output_dir, 'emergency_stop.pth'))
        print("Saved emergency checkpoint")
        raise
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()
        raise
    finally:
        trainer.cleanup_memory()


if __name__ == '__main__':
    main()
