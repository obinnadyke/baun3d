# loss_metric.py - Loss function and Metrics Computation Module (MIDL2026)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion
import warnings
warnings.filterwarnings('ignore')


class DiceLoss(nn.Module):
    """Dice Loss with proper class weighting and empty mask handling"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, class_weights=None):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        dice_loss = 0
        for c in range(1, pred.shape[1]):  # Skip background
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]

            # Handle empty masks - penalize false positives but don't crash
            if target_c.sum() == 0:
                # If ground truth is empty, penalize predictions conservatively
                loss = pred_c.mean() * 0.1  # Reduced penalty
            else:
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                loss = 1 - dice

            if class_weights is not None and c-1 < len(class_weights):
                loss *= class_weights[c-1]

            dice_loss += loss

        return dice_loss / (pred.shape[1] - 1)

class TumorSpecificLoss(nn.Module):
    """Dedicated tumor loss with SAFE gradient handling"""
    def __init__(self, config):
        super().__init__()
        self.tumor_label = config.tumor_label
        self.smooth = 1e-5
        # Conservative gradient multiplier
        self.gradient_multiplier = getattr(config, 'tumor_gradient_multiplier', 1.2)

    def forward(self, pred, target):
        tumor_pred = torch.softmax(pred, dim=1)[:, self.tumor_label].clamp(1e-6, 1-1e-6)
        tumor_target = (target == self.tumor_label).float()

        if tumor_target.sum() == 0:
            # Return zero loss but maintain gradient graph
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Standard dice loss
        intersection = (tumor_pred * tumor_target).sum()
        union = tumor_pred.sum() + tumor_target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        return loss

class BoundaryLoss(nn.Module):
    """Boundary loss using distance transforms"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, contour_map, fg_mask):
        pred_softmax = F.softmax(pred, dim=1)
        # Focus on foreground boundary
        fg_pred = pred_softmax[:, 1:].sum(dim=1, keepdim=True)
        boundary_pred = F.avg_pool3d(fg_pred, 3, stride=1, padding=1)
        boundary_pred = torch.abs(fg_pred - boundary_pred)

        # Weight by contour map
        loss = F.mse_loss(boundary_pred * fg_mask, contour_map * fg_mask)
        return loss

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for class imbalance"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target, class_weights=None):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        tversky_loss = 0
        for c in range(1, pred.shape[1]):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]

            TP = (pred_c * target_c).sum()
            FP = (pred_c * (1 - target_c)).sum()
            FN = ((1 - pred_c) * target_c).sum()

            tversky = TP / (TP + self.alpha * FP + self.beta * FN + 1e-8)
            focal_tversky = (1 - tversky) ** self.gamma

            if class_weights is not None and c-1 < len(class_weights):
                focal_tversky *= class_weights[c-1]

            tversky_loss += focal_tversky

        return tversky_loss / (pred.shape[1] - 1)

class CompositeLoss(nn.Module):
    """Composite Loss with dynamic weighting"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.ft_loss = FocalTverskyLoss()
        self.tumor_loss = TumorSpecificLoss(config)

        # Fixed weights (no dynamic changes during forward)
        self.dice_weight = config.dice_weight
        self.boundary_weight = config.boundary_weight
        self.ft_weight = config.focal_tversky_weight

        # Loss scaling to prevent explosion
        self.loss_scale = 1.0

    def forward(self, pred, target, contour_map, use_boundary=False, epoch=None):
        # Get class weights from config
        class_weights = [self.config.organ_weight, self.config.tumor_weight]

        # Compute losses with NaN/Inf guards
        dice = self.dice_loss(pred, target, class_weights)
        ft = self.ft_loss(pred, target, class_weights)
        tumor = self.tumor_loss(pred, target)

        # Clamp individual losses
        dice = torch.clamp(dice, min=0.0, max=5.0)
        ft = torch.clamp(ft, min=0.0, max=5.0)
        tumor = torch.clamp(tumor, min=0.0, max=5.0)

        base_tumor_weight = self.config.tumor_loss_weight

        # Small tumor bonus with SAFE bounds
        tumor_target = (target == self.config.tumor_label).float()
        tumor_size = tumor_target.sum().item()
        size_multiplier = 1.0

        if 0 < tumor_size < 500:
            size_multiplier = min(1.5, 500.0 / max(tumor_size, 10.0))

        # SAFE tumor loss scaling
        tumor_loss_scaled = tumor * base_tumor_weight * size_multiplier

        # GRADIENT CLIPPING INSIDE LOSS
        tumor_loss_scaled = torch.clamp(tumor_loss_scaled, max=2.0)

        # Combine with stability check
        loss = self.dice_weight * dice + self.ft_weight * ft + tumor_loss_scaled

        # Boundary loss (optional) - DELAYED ACTIVATION
        if use_boundary and contour_map is not None and epoch and epoch > 25:  # Increased from 20
            fg_mask = (target > 0).float().unsqueeze(1)
            boundary = self.boundary_loss(pred, contour_map, fg_mask)

            # Clamp boundary loss
            boundary = torch.clamp(boundary, min=0.0, max=3.0)

            # Scale boundary weight down for stability
            loss += (self.boundary_weight * 0.3) * boundary  # Reduced from 0.5

        # Final safety clamp
        loss = torch.clamp(loss, min=0.0, max=10.0)

        return loss

# Evaluation Metrics
class MetricCalculator:
    """
    Calculate comprehensive segmentation metrics for evaluation
    """

    def __init__(self, num_classes):
        """
        num_classes: Total number of segmentation classes (including background)
        """
        self.num_classes = num_classes

    # Dice Score: Overlap measure (0-1, higher is better)
    def dice_score(self, pred, target, class_idx):
        """
        Dice coefficient for specific class
        Formula: Dice = 2 * |P ∩ T| / (|P| + |T|)
        """
        pred = (pred == class_idx).astype(np.float32)
        target = (target == class_idx).astype(np.float32)

        # Handle edge case: both empty
        if target.sum() == 0:
            return 1.0 if pred.sum() == 0 else 0.0

        intersection = (pred * target).sum()
        dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
        return dice

    # IoU: Intersection over Union (Jaccard Index)
    def iou_score(self, pred, target, class_idx):
        """
        Intersection over Union (Jaccard Index)
        Formula: IoU = |P ∩ T| / |P ∪ T|
        """
        pred = (pred == class_idx).astype(np.float32)
        target = (target == class_idx).astype(np.float32)

        if target.sum() == 0:
            return 1.0 if pred.sum() == 0 else 0.0

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-8)
        return iou

    # Sensitivity (Recall): True positive rate
    def sensitivity(self, pred, target, class_idx):
        """
        Sensitivity (Recall/True Positive Rate)
        Formula: Sensitivity = TP / (TP + FN)
        """
        pred = (pred == class_idx).astype(np.float32)
        target = (target == class_idx).astype(np.float32)

        if target.sum() == 0:
            return 1.0 if pred.sum() == 0 else 0.0

        TP = (pred * target).sum()
        FN = ((1 - pred) * target).sum()
        sensitivity = TP / (TP + FN + 1e-8)
        return sensitivity

    # Specificity: True negative rate
    def specificity(self, pred, target, class_idx):
        """
        Specificity (True Negative Rate)
        Formula: Specificity = TN / (TN + FP)
        """
        pred = (pred == class_idx).astype(np.float32)
        target = (target == class_idx).astype(np.float32)

        TN = ((1 - pred) * (1 - target)).sum()
        FP = (pred * (1 - target)).sum()
        specificity = TN / (TN + FP + 1e-8)
        return specificity

    # Precision: Positive predictive value
    def precision(self, pred, target, class_idx):
        """
        Precision (Positive Predictive Value)
        Formula: Precision = TP / (TP + FP)
        """
        pred = (pred == class_idx).astype(np.float32)
        target = (target == class_idx).astype(np.float32)

        if pred.sum() == 0:
            return 1.0 if target.sum() == 0 else 0.0

        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        precision = TP / (TP + FP + 1e-8)
        return precision

    # HD95: 95th percentile Hausdorff Distance (mm)
    def hausdorff_distance_95(self, pred, target, class_idx, spacing=(1.0, 1.0, 1.0)):
        """
        Hausdorff Distance 95th percentile
        spacing: Voxel spacing in mm (z, y, x)
        """
        pred = (pred == class_idx).astype(np.uint8)
        target = (target == class_idx).astype(np.uint8)

        # Handle empty cases
        if pred.sum() == 0 or target.sum() == 0:
            return 0.0 if pred.sum() == target.sum() else 100.0

        # Extract surface points using morphological erosion
        pred_surface = pred ^ binary_erosion(pred)
        target_surface = target ^ binary_erosion(target)

        pred_points = np.argwhere(pred_surface)
        target_points = np.argwhere(target_surface)

        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0

        # Scale points by voxel spacing to get real-world coordinates
        pred_points = pred_points * spacing
        target_points = target_points * spacing

        # Compute pairwise distances between all surface points
        from scipy.spatial.distance import cdist
        distances_pred_to_target = cdist(pred_points, target_points).min(axis=1)
        distances_target_to_pred = cdist(target_points, pred_points).min(axis=1)

        # Combine both directions and take 95th percentile
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        hd95 = np.percentile(all_distances, 95)

        return hd95

    # ASD: Average Symmetric Surface Distance (mm)
    def average_surface_distance(self, pred, target, class_idx, spacing=(1.0, 1.0, 1.0)):
        """
        Average Symmetric Surface Distance
        """
        pred = (pred == class_idx).astype(np.uint8)
        target = (target == class_idx).astype(np.uint8)

        # Handle empty cases
        if pred.sum() == 0 or target.sum() == 0:
            return 0.0 if pred.sum() == target.sum() else 100.0

        # Extract surface points
        pred_surface = pred ^ binary_erosion(pred)
        target_surface = target ^ binary_erosion(target)

        pred_points = np.argwhere(pred_surface)
        target_points = np.argwhere(target_surface)

        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0

        # Scale by voxel spacing
        pred_points = pred_points * spacing
        target_points = target_points * spacing

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances_pred_to_target = cdist(pred_points, target_points).min(axis=1)
        distances_target_to_pred = cdist(target_points, pred_points).min(axis=1)

        # Compute average distance in both directions
        asd = (distances_pred_to_target.sum() + distances_target_to_pred.sum()) / \
              (len(distances_pred_to_target) + len(distances_target_to_pred))

        return asd

    def compute_all_metrics(self, pred, target, organ_label, tumor_label, spacing=(1.0, 1.0, 1.0), fast_mode=True):
        """
        Compute comprehensive metrics for segmentation evaluation
        """
        metrics = {}

        # Fast metrics (always computed) - used during training
        metrics['organ_dice'] = self.dice_score(pred, target, organ_label)
        metrics['tumor_dice'] = self.dice_score(pred, target, tumor_label)
        metrics['avg_dice'] = (metrics['organ_dice'] + metrics['tumor_dice']) / 2

        # Slow metrics (only for final evaluation) - used for validation/testing
        if not fast_mode:
            # Organ metrics
            metrics['organ_iou'] = self.iou_score(pred, target, organ_label)
            metrics['organ_sensitivity'] = self.sensitivity(pred, target, organ_label)
            metrics['organ_specificity'] = self.specificity(pred, target, organ_label)
            metrics['organ_precision'] = self.precision(pred, target, organ_label)
            metrics['organ_hd95'] = self.hausdorff_distance_95(pred, target, organ_label, spacing)
            metrics['organ_asd'] = self.average_surface_distance(pred, target, organ_label, spacing)

            # Tumor metrics
            metrics['tumor_iou'] = self.iou_score(pred, target, tumor_label)
            metrics['tumor_sensitivity'] = self.sensitivity(pred, target, tumor_label)
            metrics['tumor_specificity'] = self.specificity(pred, target, tumor_label)
            metrics['tumor_precision'] = self.precision(pred, target, tumor_label)
            metrics['tumor_hd95'] = self.hausdorff_distance_95(pred, target, tumor_label, spacing)
            metrics['tumor_asd'] = self.average_surface_distance(pred, target, tumor_label, spacing)

        return metrics


def get_loss_function(config):
    """
    Factory function to get composite loss function
    """
    return CompositeLoss(config)


def get_metric_calculator(config):
    """
    Factory function to get metric calculator
    """
    return MetricCalculator(config.num_classes)
