# plotter.py - Complete Visualization Module for BAUN3D (MIDL2026)

import os
import sys
import traceback
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion
from tqdm import tqdm
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json
import gc

# Check for OpenCV
try:
    import cv2
except ImportError:
    cv2 = None
    print("WARNING: OpenCV not found. Install with: pip install opencv-python-headless")

warnings.filterwarnings('ignore')


class DatasetAnalyzer:
    """Pre-training dataset stat analysis"""
    def __init__(self, config, data_dir, output_dir):
        self.config = config
        self.data_dir = os.path.join(data_dir, config.dataset_name)
        self.output_dir = os.path.join(output_dir, 'dataset_analysis')
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_dir = os.path.join(self.data_dir, 'imagesTr')
        self.label_dir = os.path.join(self.data_dir, 'labelsTr')

        if not os.path.isdir(self.image_dir):
            print(f"[ANALYSIS] Image directory not found: {self.image_dir}")
            self.image_files = []
            return

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith('.nii') or f.endswith('.nii.gz')
        ])

    def collect_statistics(self):
        if len(self.image_files) == 0:
            print("[ANALYSIS] No images found, skipping statistics")
            return None

        stats = {
            'intensities': [],
            'organ_volumes': [],
            'tumor_volumes': [],
            'tumor_sizes': [],
            'organ_present': 0,
            'tumor_present': 0
        }

        print(f"Analyzing {len(self.image_files)} cases...")
        for img_file in tqdm(self.image_files, desc='Collecting statistics'):
            img_path = os.path.join(self.image_dir, img_file)
            lbl_path = os.path.join(self.label_dir, img_file)

            try:
                img_nii = nib.load(img_path)
                lbl_nii = nib.load(lbl_path)

                image = img_nii.get_fdata()
                label = lbl_nii.get_fdata()

                stats['intensities'].extend(image.flatten()[::1000].tolist())

                organ_mask = (label == self.config.organ_label)
                tumor_mask = (label == self.config.tumor_label)

                ov = int(organ_mask.sum())
                tv = int(tumor_mask.sum())

                stats['organ_volumes'].append(ov)
                stats['tumor_volumes'].append(tv)

                if ov > 0:
                    stats['organ_present'] += 1
                if tv > 0:
                    stats['tumor_present'] += 1
                    stats['tumor_sizes'].append(tv)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

        return stats

    def generate_plots(self, stats):
        if stats is None:
            return

        try:
            # Save text summary
            with open(os.path.join(self.output_dir, 'statistics_summary.txt'), 'w') as f:
                f.write(f"Dataset: {self.config.dataset_name.upper()}\n")
                f.write(f"Total cases: {len(self.image_files)}\n")
                f.write(f"Organ present: {stats['organ_present']}\n")
                f.write(f"Tumor present: {stats['tumor_present']}\n")
                f.write(f"Mean organ volume: {np.mean(stats['organ_volumes']):.0f} voxels\n")
                if len(stats['tumor_sizes']) > 0:
                    f.write(f"Mean tumor volume: {np.mean(stats['tumor_sizes']):.0f} voxels\n")

            # Plot 1: Intensity distribution
            fig1 = plt.figure(figsize=(10, 6))
            intensities = np.array(stats['intensities'])
            intensities = intensities[~np.isnan(intensities)]
            plt.hist(intensities, bins=100, alpha=0.7, edgecolor='black')
            plt.xlabel('Intensity (HU)')
            plt.ylabel('Frequency')
            plt.title(f'{self.config.dataset_name.upper()} - CT Intensity Distribution')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'intensity_distribution.png'), dpi=150)
            plt.close(fig1)

            # Plot 2: Volume distribution
            fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(stats['organ_volumes'], bins=30, alpha=0.7, edgecolor='black')
            axes[0].set_title('Organ Volume')
            axes[0].set_xlabel('Voxels')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(alpha=0.3)

            tv = [v for v in stats['tumor_volumes'] if v > 0]
            if tv:
                axes[1].hist(tv, bins=30, alpha=0.7, edgecolor='black', color='orange')
                axes[1].set_title('Tumor Volume')
                axes[1].set_xlabel('Voxels')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'volume_distribution.png'), dpi=150)
            plt.close(fig2)

            # Plot 3: Class distribution
            fig3, ax = plt.subplots(figsize=(8, 6))
            categories = ['Organ Present', 'Tumor Present']
            counts = [stats['organ_present'], stats['tumor_present']]
            bars = ax.bar(categories, counts, alpha=0.85, edgecolor='black', color=['red', 'green'])
            ax.set_ylabel('Number of Cases')
            ax.set_title('Class Distribution')
            ax.grid(axis='y', alpha=0.3)

            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                       f'{int(count)}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'), dpi=150)
            plt.close(fig3)

            # Plot 4: Tumor size analysis
            if len(stats['tumor_sizes']) > 0:
                ts = np.array(stats['tumor_sizes'])
                fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

                axes[0].hist(ts, bins=50, alpha=0.7, edgecolor='black', color='red')
                axes[0].set_yscale('log')
                axes[0].set_title('Tumor Size Distribution (Log Scale)')
                axes[0].set_xlabel('Voxels')
                axes[0].set_ylabel('Frequency (log)')
                axes[0].grid(alpha=0.3)

                axes[1].boxplot(ts, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightcoral', alpha=0.7),
                              medianprops=dict(color='darkred', linewidth=2))
                axes[1].set_title('Tumor Size Statistics')
                axes[1].set_ylabel('Voxels')
                axes[1].grid(alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'tumor_size_analysis.png'), dpi=150)
                plt.close(fig4)

            # Close all figures and cleanup
            plt.close('all')
            gc.collect()

            print(f"Dataset analysis complete. Results saved to: {self.output_dir}")
            print(f"  - statistics_summary.txt")
            print(f"  - intensity_distribution.png")
            print(f"  - volume_distribution.png")
            print(f"  - class_distribution.png")
            if len(stats['tumor_sizes']) > 0:
                print(f"  - tumor_size_analysis.png")

        except Exception as e:
            print(f"Error generating plots: {e}")
            traceback.print_exc()
        finally:
            plt.close('all')
            gc.collect()


#-------Functions for MULTI-VIEW VISUALIZATION of SEGMENTATION Learning--------

# Helper Functions (Defined First)
def _safe_normalize(volume):
    """Normalize volume to [0,1] range safely"""
    volume = np.squeeze(volume)
    if volume.size == 0: return volume
    v_min, v_max = np.percentile(volume[volume > 0], [1, 99]) if (volume > 0).any() else (0, 1)
    if v_max > v_min:
        return np.clip((volume - v_min) / (v_max - v_min), 0, 1)
    return np.zeros_like(volume)

def _ensure_3d(volume):
    """Ensure volume is 3D (D, H, W)"""
    volume = np.asarray(volume)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim < 3: raise ValueError(f"Volume shape {volume.shape} invalid")
    return volume

def _resize_slice(slice_arr, target_size=(256, 256), is_mask=False):
    """
    Force-resize a 2D slice to a square target size to prevent aspect ratio distortion.
    """
    if cv2 is None: return slice_arr

    # INTER_NEAREST for masks (preserves integers), INTER_LINEAR for images
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(slice_arr, target_size, interpolation=interp)

def select_max_tumor_slice(gt_vol, organ_label=1, tumor_label=2):
    """
    Finds the slice index with the MAXIMUM number of tumor pixels.
    """
    # 1. Prioritize Tumor
    mask = (gt_vol == tumor_label)

    # 2. Fallback to Organ
    if mask.sum() == 0:
        mask = (gt_vol == organ_label)

    # 3. Fallback to Middle
    if mask.sum() == 0:
        return gt_vol.shape[0]//2, gt_vol.shape[1]//2, gt_vol.shape[2]//2

    # Sum along axes to find "thickest" part
    sum_x = mask.sum(axis=(1, 2))
    cx = np.argmax(sum_x)

    sum_y = mask.sum(axis=(0, 2))
    cy = np.argmax(sum_y)

    sum_z = mask.sum(axis=(0, 1))
    cz = np.argmax(sum_z)

    return cx, cy, cz

def _boundary_map(mask):
    if mask.sum() == 0: return np.zeros_like(mask, dtype=np.float32)
    return (binary_erosion(mask) ^ mask).astype(np.float32)

def _create_boundary_overlay(image_slice, seg_slice, organ_label, tumor_label):
    """Draws boundaries on the image slice using OpenCV. Preserves the CT background."""
    if cv2 is None: return np.stack([image_slice]*3, axis=-1)

    # Ensure input is correct range/type for visualization
    # Image slice should be [0,1] float -> convert to [0,255] uint8
    img_display = (image_slice * 255).astype(np.uint8)
    img_display = np.stack([img_display] * 3, axis=-1) # Grayscale to RGB

    # Draw Organ (Red)
    if (seg_slice == organ_label).any():
        mask_uint8 = (seg_slice == organ_label).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_display, cnts, -1, (255, 0, 0), 1)

    # Draw Tumor (Green)
    if (seg_slice == tumor_label).any():
        mask_uint8 = (seg_slice == tumor_label).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_display, cnts, -1, (0, 255, 0), 1)

    return img_display

# --- Main Visualization Function ---
def visualize_segmentation_comparison_multiview(
    volume, gt_volume, pred_volume,
    organ_label=1, tumor_label=2,
    metrics: dict = None,
    out_path: str = 'comparison_multiview.png',
    mode: str = 'boundary',
    title: str = None,
):
    """
    Generates a 3x3 comparison plot
    1. Resizes all slices to 256x256
    2. Handles memory layout for OpenCV
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        volume = _safe_normalize(_ensure_3d(volume))
        gt_volume = _ensure_3d(gt_volume)
        pred_volume = _ensure_3d(pred_volume)

        # --- SELECT BEST SLICES ---
        cx, cy, cz = select_max_tumor_slice(gt_volume, organ_label, tumor_label)

        # --- EXTRACT & RESIZE VIEWS ---
        # Target display resolution
        DISP_SIZE = (256, 256)

        def process_slice(slc, is_mask=False):
            # 1. Rotate 90 degrees
            rotated = np.rot90(slc)
            # 2. Ensure memory layout
            contiguous = np.ascontiguousarray(rotated)
            # 3. Resize to square
            resized = _resize_slice(contiguous, DISP_SIZE, is_mask)
            return resized

        # Row 1: AXIAL (Slice Z)
        ax_img  = process_slice(volume[:, :, cz], False)
        ax_gt   = process_slice(gt_volume[:, :, cz], True)
        ax_pred = process_slice(pred_volume[:, :, cz], True)

        # Row 2: CORONAL (Slice Y)
        cor_img  = process_slice(volume[:, cy, :], False)
        cor_gt   = process_slice(gt_volume[:, cy, :], True)
        cor_pred = process_slice(pred_volume[:, cy, :], True)

        # Row 3: SAGITTAL (Slice X)
        sag_img  = process_slice(volume[cx, :, :], False)
        sag_gt   = process_slice(gt_volume[cx, :, :], True)
        sag_pred = process_slice(pred_volume[cx, :, :], True)

        # Setup 3 rows, 3 columns Figure (with minimal spacing, tight layout)
        fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=150)
        plt.subplots_adjust(wspace=0.02, hspace=0.08, left=0.01, right=0.93, bottom=0.01, top=0.95)

        if title: fig.suptitle(title, fontsize=14, fontweight='normal', y=0.98)

        # Helper to plot
        def plot_cell(ax, img, gt, pred, col_type, row_name):
            ax.axis('off')

            if col_type == 'GT':
                overlay = _create_boundary_overlay(img, gt, organ_label, tumor_label)
                ax.imshow(overlay, aspect='equal')
                if row_name: ax.set_title(f"{row_name} - GT", fontsize=10, fontweight='normal', pad=4)

            elif col_type == 'Pred':
                overlay = _create_boundary_overlay(img, pred, organ_label, tumor_label)
                ax.imshow(overlay, aspect='equal')
                if row_name: ax.set_title(f"{row_name} - Pred", fontsize=10, fontweight='normal', pad=4)

            elif col_type == 'Heat':
                # Compute heatmap -> Recalculate boundaries on the RESIZED masks
                gt_b = _boundary_map((gt == organ_label) | (gt == tumor_label))
                pred_b = _boundary_map((pred == organ_label) | (pred == tumor_label))
                heatmap = np.abs(gt_b - pred_b)
                # Setting aspect='equal' is crucial for artifact removal
                im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1, aspect='auto')

                # We use the previous colorbar placement logic here
                plt.colorbar(im, ax=ax, fraction=0.04, pad=0.08, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

                if row_name: ax.set_title(f"{row_name} - Heat", fontsize=10, fontweight='normal', pad=4)
                return im

        # Row 1
        plot_cell(axes[0,0], ax_img, ax_gt, ax_pred, 'GT', 'AXIAL')
        plot_cell(axes[0,1], ax_img, ax_gt, ax_pred, 'Pred', 'AXIAL')
        plot_cell(axes[0,2], ax_img, ax_gt, ax_pred, 'Heat', 'AXIAL')

        # Row 2
        plot_cell(axes[1,0], cor_img, cor_gt, cor_pred, 'GT', 'CORONAL')
        plot_cell(axes[1,1], cor_img, cor_gt, cor_pred, 'Pred', 'CORONAL')
        plot_cell(axes[1,2], cor_img, cor_gt, cor_pred, 'Heat', 'CORONAL')

        # Row 3
        plot_cell(axes[2,0], sag_img, sag_gt, sag_pred, 'GT', 'SAGITTAL')
        plot_cell(axes[2,1], sag_img, sag_gt, sag_pred, 'Pred', 'SAGITTAL')
        plot_cell(axes[2,2], sag_img, sag_gt, sag_pred, 'Heat', 'SAGITTAL')

        plt.savefig(out_path, dpi=150) # Removed bbox_inches='tight' to respect subplots_adjust
        plt.close(fig)
        print(f"[VIZ] Saved visualization to {out_path}")
        return True

    except Exception as e:
        print(f"[VIZ ERROR] {e}")
        traceback.print_exc()
        return False
    finally:
        plt.close('all')
        gc.collect()

# (Optional) Extended Comparative Segmentation Visualizer
def visualize_segmentation_comparison(
    image_slice, gt_slice, pred_slice,
    organ_label=1, tumor_label=2,
    metrics: dict = None,
    out_path: str = 'comparison.png',
    mode: str = 'boundary',
    title: str = None,
):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig = plt.figure(figsize=(10, 6), facecolor='black')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1])
        axC = fig.add_subplot(gs[2])

        # Simple GT/Pred/Error view for single slices
        axA.imshow(image_slice, cmap='gray', vmin=0, vmax=1)
        axA.contour(gt_slice == organ_label, colors='red', linewidths=1)
        axA.set_title('GT', color='white')
        axA.axis('off')

        axB.imshow(image_slice, cmap='gray', vmin=0, vmax=1)
        axB.contour(pred_slice == organ_label, colors='red', linewidths=1)
        axB.set_title('PRED', color='white')
        axB.axis('off')

        # Helper function for visualization purposes
        def _compute_masked_heatmap(gt, pred):
            # Placeholder/simplified heatmap computation
            if gt.shape != pred.shape: return None
            return np.abs(gt.astype(float) - pred.astype(float))

        heat = _compute_masked_heatmap(gt_slice>0, pred_slice>0)
        if heat is not None:
             axC.imshow(heat, cmap='jet')
        axC.set_title('ERR', color='white')
        axC.axis('off')

        plt.tight_layout()
        plt.savefig(out_path, dpi=100, facecolor='black')
        plt.close(fig)
        return True

    except Exception as e:
        print(f"[VIZ] visualize_segmentation_comparison failed: {e}")
        traceback.print_exc()
        return False
    finally:
        plt.close('all')
        gc.collect()

#-------------------------------------------------------------------------------

# Training Progress Visualization
class TrainingPlotter:
    """Plot training curves showing model learning progress, saving each metric as a single figure."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_curves(self, history, organ_name='Organ'):
        """Generates and saves nine separate plots for training progression."""

        # Helper function to generate and save a single plot
        def _save_single_plot(title, filename, plot_fn, x_data=None):
            fig, ax = plt.subplots(figsize=(10, 6))
            if plot_fn(ax, x_data):
                ax.set_title(title, fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Epoch', fontsize=12)

                plt.tight_layout()
                save_path = os.path.join(self.output_dir, filename)
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"[PLOT] Saved {title} to: {save_path}")
                return True
            else:
                plt.close(fig)
                return False

        try:
            # Use the longest history list for alignment
            epochs_loss = list(range(1, len(history.get('train_loss', [])) + 1))
            epochs_dice = list(range(1, len(history.get('val_organ_dice', [])) + 1))
            epochs = epochs_loss # Baseline for scheduling plots

            # --- 1. Training Loss (Train vs Val) ---
            def plot_loss(ax, x_data):
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                if not (train_loss and val_loss): return False
                ax.plot(x_data, train_loss, 'b-', label='Train Loss', linewidth=2)
                ax.plot(epochs_dice, val_loss, 'r-', label='Val Loss', linewidth=2)
                ax.set_ylabel('Loss', fontsize=12); ax.legend()
                return True
            _save_single_plot('Training Loss (Train vs Val)', '1_loss_curves.png', plot_loss, epochs)

            # --- 2. Organ Dice Score (Train vs Val) ---
            def plot_organ_dice(ax, x_data):
                train_organ = history.get('train_organ_dice', [])
                val_organ = history.get('val_organ_dice', [])
                if not (train_organ and val_organ): return False
                ax.plot(epochs_dice, train_organ, 'b-', label=f'Train {organ_name}', linewidth=2)
                ax.plot(epochs_dice, val_organ, 'r-', label=f'Val {organ_name}', linewidth=2)
                ax.set_ylabel('Dice', fontsize=12); ax.set_ylim([0, 1]); ax.legend()
                return True
            _save_single_plot(f'{organ_name} Dice Score (Train vs Val)', '2_organ_dice.png', plot_organ_dice, epochs_dice)

            # --- 3. Tumor Dice Score (Train vs Val) ---
            def plot_tumor_dice(ax, x_data):
                train_tumor = history.get('train_tumor_dice', [])
                val_tumor = history.get('val_tumor_dice', [])
                if not (train_tumor and val_tumor): return False
                ax.plot(epochs_dice, train_tumor, 'b-', label='Train Tumor', linewidth=2)
                ax.plot(epochs_dice, val_tumor, 'r-', label='Val Tumor', linewidth=2)
                ax.set_ylabel('Dice', fontsize=12); ax.set_ylim([0, 1]); ax.legend()
                return True
            _save_single_plot('Tumor Dice Score (Train vs Val)', '3_tumor_dice.png', plot_tumor_dice, epochs_dice)

            # --- 4. Average Dice Score (Train vs Val) ---
            def plot_avg_dice(ax, x_data):
                train_avg = history.get('train_avg_dice', [])
                val_avg = history.get('val_avg_dice', [])
                if not (train_avg and val_avg): return False
                ax.plot(epochs_dice, train_avg, 'b-', label='Train Avg', linewidth=2)
                ax.plot(epochs_dice, val_avg, 'r-', label='Val Avg', linewidth=2)
                ax.set_ylabel('Dice', fontsize=12); ax.set_ylim([0, 1]); ax.legend()
                return True
            _save_single_plot('Average Dice Score (Train vs Val)', '4_avg_dice.png', plot_avg_dice, epochs_dice)

            # --- 5. Learning Rate Schedule ---
            def plot_lr(ax, x_data):
                lr = history.get('learning_rate')
                if not lr: return False
                ax.plot(epochs, lr, 'g-', linewidth=2)
                ax.set_ylabel('LR', fontsize=12)
                return True
            _save_single_plot('Learning Rate Schedule', '5_lr_schedule.png', plot_lr, epochs)

            # --- 6. Boundary Strength Schedule ---
            def plot_boundary_strength(ax, x_data):
                strength = history.get('boundary_strength')
                if not strength: return False
                ax.plot(epochs, strength, 'purple', linewidth=2)
                ax.set_ylabel('Boundary Strength', fontsize=12); ax.set_ylim([0, 1])
                return True
            _save_single_plot('Boundary Refinement Schedule', '6_boundary_strength.png', plot_boundary_strength, epochs)

            # --- 7. Curriculum Weight Schedule ---
            def plot_curriculum_weight(ax, x_data):
                weight = history.get('curriculum_weight')
                if not weight: return False
                ax.plot(epochs, weight, 'orange', linewidth=2)
                ax.set_ylabel('Tumor Weight', fontsize=12); ax.set_ylim([0, 1])
                return True
            _save_single_plot('Curriculum Learning Schedule', '7_curriculum_weight.png', plot_curriculum_weight, epochs)

            # --- 8. Gradient Norm Monitor ---
            def plot_grad_norm(ax, x_data):
                grad_norm = history.get('grad_norm')
                if not grad_norm: return False
                ax.plot(epochs, grad_norm, 'brown', linewidth=2)
                ax.set_ylabel('Gradient Norm', fontsize=12)
                return True
            _save_single_plot('Gradient Norm Monitor', '8_grad_norm.png', plot_grad_norm, epochs)

            # --- 9. Overfitting Gap ---
            def plot_gap(ax, x_data):
                train_avg = history.get('train_avg_dice', [])
                val_avg = history.get('val_avg_dice', [])
                if not (train_avg and val_avg and len(val_avg) > 1): return False

                # Interpolate train dice to match the frequency of val dice
                train_interp = np.interp(
                    np.linspace(1, len(x_data), len(val_avg)),
                    np.linspace(1, len(x_data), len(train_avg)),
                    train_avg
                )
                gap = np.array(train_interp) - np.array(val_avg)

                ax.plot(np.linspace(1, len(x_data), len(gap)), gap, 'red', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.fill_between(np.linspace(1, len(x_data), len(gap)),
                                 gap, 0, alpha=0.3, color='red')
                ax.set_ylabel('Train - Val Dice', fontsize=12)
                ax.set_ylim([np.min(gap)-0.05, np.max(gap)+0.05])
                return True
            _save_single_plot('Overfitting Gap', '9_overfitting_gap.png', plot_gap, epochs)

            print("\n[PLOT] All single training curve plots generated successfully.")
            return True

        except Exception as e:
            print(f"[PLOT] Training curve plotting failed: {str(e)}")
            traceback.print_exc()
            return False

        finally:
            # Ensure all figures are closed
            plt.close('all')
            gc.collect()


# Post-Training Analysis
# In plotter.py (as a standalone function)
def plot_confusion_matrix(y_true, y_pred, class_names, output_path, dataset_name=''):
    """
    Plot confusion matrix for segmentation classes
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(len(class_names)))

    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='normal')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='normal')

    plt.suptitle(f'{dataset_name.upper()} - Segmentation Confusion Matrix',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return cm

#------------------------------------------------------------------------------

# Top-level API Functions
def analyze_dataset(config, data_dir, output_dir):
    try:
        analyzer = DatasetAnalyzer(config, data_dir, output_dir)
        if len(analyzer.image_files) > 0:
            stats = analyzer.collect_statistics()
            analyzer.generate_plots(stats)
        else:
            print("[ANALYSIS] No data found, skipping")
        return True
    except Exception as e:
        print(f"[ANALYSIS] Failed: {e}")
        traceback.print_exc()
        return False
    finally:
        plt.close('all')
        gc.collect()


def plot_training_curves(history, output_dir, organ_name='Organ'):
    """Plot training progress curves"""
    try:
        out_dir = os.path.join(output_dir, 'training_plots')
        plotter = TrainingPlotter(out_dir)
        return plotter.plot_training_curves(history, organ_name=organ_name)
    except Exception as e:
        print(f"[PLOT] Training curves plotting failed: {e}")
        traceback.print_exc()
        return False
    finally:
        plt.close('all')
        gc.collect()


def generate_post_training_analysis(config, output_dir, predictions=None):
    """Generate all post-training analysis"""
    try:
        print("Generating Post-Training Analysis...")

        analysis_dir = os.path.join(output_dir, 'post_training_analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        if predictions is not None and predictions['y_true'] is not None:
            class_names = ['Background', 'Organ', 'Tumor']
            output_path = os.path.join(analysis_dir, 'confusion_matrix.png')

            plot_confusion_matrix(
                predictions['y_true'],
                predictions['y_pred'],
                class_names,
                output_path,
                dataset_name=config.dataset_name
            )
            print(f"[ANALYSIS] Confusion Matrix saved to {output_path}")
        else:
            print("[ANALYSIS] Skipping Confusion Matrix: Prediction data not provided to function.")

        # Placeholder for future analysis
        print("Post-training analysis completed")
        return True

    except Exception as e:
        print(f"[ANALYSIS] Post-training analysis failed: {e}")
        traceback.print_exc()
        return False
    finally:
        plt.close('all')
        gc.collect()
