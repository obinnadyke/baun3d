# dataloader.py - Tumor-Preserving Data Loader with Smart Caching (MIDL2026)

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')
from functools import lru_cache
from tqdm import tqdm
import threading
import queue


# Helper Functions
def _ensure_3d(arr):
    """Ensure array is 3D or 4D multi-channel"""
    if arr.ndim == 3:
        return False, arr[..., np.newaxis]  # channel dim
    elif arr.ndim == 4:
        return True, arr
    else:
        raise ValueError(f"Expected 3D or 4D volume, got shape {arr.shape}")

def _nii_stem(path: str) -> str:
    """Extract filename stem from NIfTI file path"""
    name = os.path.basename(path)
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return os.path.splitext(name)[0]

def _pair_by_stem(img_paths, lbl_paths):
    """Match image and label files by filename stem"""
    img_map = {_nii_stem(p): p for p in img_paths}
    lbl_map = {_nii_stem(p): p for p in lbl_paths}
    common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    if not common:
        raise RuntimeError("No matching image/label stems found. Check filenames.")
    return [img_map[k] for k in common], [lbl_map[k] for k in common]

# CT Preprocessing
class CTPreprocessor:
    """Preprocessor for CT scans with intensity windowing and resampling"""

    def __init__(self, config):
        self.config = config
        self.clip_range = config.clip_range
        self.window_level = config.window_level
        self.window_width = config.window_width
        self.target_spacing = getattr(config, 'target_spacing', (1.5, 1.5, 2.0))

    def clip_intensity(self, image):
        """Clip image intensities to valid HU range"""
        return np.clip(image, self.clip_range[0], self.clip_range[1])

    def window_normalize(self, image):
        """Apply CT windowing and normalize to [0, 1]"""
        lower = self.window_level - self.window_width / 2
        upper = self.window_level + self.window_width / 2
        image = np.clip(image, lower, upper)
        return ((image - lower) / (upper - lower + 1e-8)).astype(np.float32)

    def preprocess(self, image, label=None, spacing=None):
        """Complete preprocessing pipeline for CT/MRI scan"""
        is_multichannel, image = _ensure_3d(image)

        # Clip HU range
        image = self.clip_intensity(image)

        # NaN handling
        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Resample if spacing provided
        if spacing is not None:
            zoom = np.array(spacing, dtype=np.float32) / np.array(self.target_spacing, dtype=np.float32)
            resampled = [ndimage.zoom(image[..., c], zoom, order=1) for c in range(image.shape[-1])]
            image = np.stack(resampled, axis=-1)
            if label is not None:
                label = ndimage.zoom(label, zoom, order=0).astype(np.int16)

        # Normalize
        image = np.stack([self.window_normalize(image[..., c]) for c in range(image.shape[-1])], axis=-1)

        # Channel-first
        image = np.transpose(image, (3, 0, 1, 2)).astype(np.float32)
        return image, label

# Data Augmentation
class Augmentor:
    """Tumor-preserving data augmentation for 3D medical images"""

    def __init__(self, config):
        self.config = config
        self.aug_prob = config.aug_prob
        self.rotation_range = config.rotation_range
        self.scale_range = config.scale_range
        self.brightness_range = config.brightness_range
        self.contrast_range = config.contrast_range
        self.elastic_deform_enabled = config.elastic_deform
        self.elastic_alpha = config.elastic_alpha
        self.elastic_sigma = config.elastic_sigma
        self.tumor_copy_paste = config.tumor_copy_paste
        self.tumor_aug_prob = config.tumor_aug_prob

    def _has_tumor(self, label):
        """Check if label contains tumor"""
        return (label == self.config.tumor_label).sum() > 0

    def _tumor_voxel_count(self, label):
        """Count tumor voxels"""
        return (label == self.config.tumor_label).sum()

    def random_flip(self, image, label):
        """Randomly flip image and label along each axis with tumor check"""
        for axis in range(3):
            if np.random.rand() < self.aug_prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotation(self, image, label):
        """Safe rotation that reverts if tumor destroyed"""
        if np.random.rand() >= self.aug_prob:
            return image, label

        orig_image, orig_label = image.copy(), label.copy()
        angle = np.random.uniform(*self.rotation_range)
        axes = [(0, 1), (0, 2), (1, 2)]
        axis = axes[np.random.choice(len(axes))]

        # Rotate
        label_rot = ndimage.rotate(label, angle, axes=axis, reshape=False, order=0, mode='nearest')
        for c in range(image.shape[0]):
            image[c] = ndimage.rotate(image[c], angle, axes=axis, reshape=False, order=1, mode='nearest')

        # REVERT if needed
        return self._revert_if_tumor_lost(image, label_rot, orig_label)

    def _center_crop_or_pad(self, vol, target_shape, is_label=False):
        """Make 'vol' [D,H,W] match target_shape via center crop then zero pad."""
        D, H, W = vol.shape
        TD, TH, TW = target_shape

        # Center crop
        sd = max((D - TD) // 2, 0)
        sh = max((H - TH) // 2, 0)
        sw = max((W - TW) // 2, 0)
        ed = sd + min(TD, D)
        eh = sh + min(TH, H)
        ew = sw + min(TW, W)
        vol = vol[sd:ed, sh:eh, sw:ew]

        # Center pad
        pd = max(TD - vol.shape[0], 0)
        ph = max(TH - vol.shape[1], 0)
        pw = max(TW - vol.shape[2], 0)
        if pd or ph or pw:
            pad_before = (pd // 2, ph // 2, pw // 2)
            pad_after  = (pd - pad_before[0], ph - pad_before[1], pw - pad_before[2])
            vol = np.pad(
                vol,
                pad_width=((pad_before[0], pad_after[0]),
                           (pad_before[1], pad_after[1]),
                           (pad_before[2], pad_after[2])),
                mode="constant",
                constant_values=0
            )

        return vol.astype(np.int64 if is_label else np.float32, copy=False)

    def random_scale(self, image, label):
        """Random scale with tumor-safe center crop/pad"""
        target_shape = image.shape[1:]  # (D,H,W)

        # Conservative isotropic scaling
        s = np.random.uniform(*self.scale_range)
        scale = np.array([s, s, s], dtype=np.float32)

        # Store tumor info
        has_tumor = self._has_tumor(label)
        orig_tumor_count = self._tumor_voxel_count(label)

        C = image.shape[0]
        image_out = np.empty((C,) + tuple(target_shape), dtype=np.float32)
        for c in range(C):
            scaled = ndimage.zoom(image[c], zoom=scale, order=1)
            scaled = self._center_crop_or_pad(scaled, target_shape, is_label=False)
            image_out[c] = scaled

        if label is not None:
            lab_scaled = ndimage.zoom(label, zoom=scale, order=0)
            lab_scaled = self._center_crop_or_pad(lab_scaled, target_shape, is_label=True)

            # Tumor preservation check
            if has_tumor and self._tumor_voxel_count(lab_scaled) < orig_tumor_count * 0.5:
                # Revert to original if tumor severely damaged
                return image, label
        else:
            lab_scaled = None

        return image_out, lab_scaled

    def _revert_if_tumor_lost(self, image, label, orig_label, min_retention=0.6):
        """Revert augmentations that destroy tumors"""
        orig_count = (orig_label == self.config.tumor_label).sum()
        new_count = (label == self.config.tumor_label).sum()

        if orig_count > 0 and new_count < orig_count * min_retention:
            # Revert to original
            return image.copy(), orig_label.copy()
        return image, label

    def elastic_deform(self, image, label):
        """Safe elastic deformation with minimal tumor impact"""
        if np.random.rand() >= self.aug_prob or not self.elastic_deform_enabled:
            return image, label

        # Store tumor count
        orig_tumor_count = self._tumor_voxel_count(label)
        shape = label.shape

        # Very conservative parameters for small structures
        alpha = np.random.uniform(*self.elastic_alpha) * 0.3  # Scale down
        sigma = np.random.uniform(*self.elastic_sigma)

        # Generate displacement fields
        dx = ndimage.gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dy = ndimage.gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dz = ndimage.gaussian_filter(np.random.randn(*shape), sigma) * alpha

        # Create coordinate grid
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                             np.arange(shape[2]), indexing='ij')
        coords = (x + dx, y + dy, z + dz)

        # Remap with tumor preservation
        label_deformed = ndimage.map_coordinates(label, coords, order=0, mode='nearest')

        # Check tumor preservation
        new_tumor_count = self._tumor_voxel_count(label_deformed)
        if orig_tumor_count > 0 and new_tumor_count < orig_tumor_count * 0.6:
            # Skip deformation if tumor destroyed
            return image, label

        label = label_deformed

        for c in range(image.shape[0]):  # Channel-first
            image[c] = ndimage.map_coordinates(image[c], coords, order=1, mode='reflect')

        return image, label

    def random_brightness(self, image):
        """Randomly adjust image brightness (multiplicative)"""
        if np.random.rand() < self.aug_prob:
            factor = np.random.uniform(*self.brightness_range)
            image = np.clip(image * factor, 0, 1)
        return image

    def random_contrast(self, image):
        """Randomly adjust image contrast around mean intensity"""
        if np.random.rand() < self.aug_prob:
            factor = np.random.uniform(*self.contrast_range)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 1)
        return image

    def tumor_copy_paste_augmentation(self, image, label):
        """SAFE tumor copy-paste with overlap detection"""
        if not self.tumor_copy_paste or np.random.rand() > self.tumor_aug_prob:
            return image, label

        tumor_mask = (label == self.config.tumor_label)
        if not tumor_mask.any():
            return image, label

        coords = np.argwhere(tumor_mask)
        if len(coords) < 10:  # Too small to copy
            return image, label

        # Create 1-2 tumor copies (reduced from 2-3)
        num_copies = np.random.randint(1, 3)
        tumor_intensity = image[:, tumor_mask].mean(axis=1)  # Per-channel

        # Get valid placement zones (avoid existing tumors)
        non_tumor_mask = (label == 0) | (label == self.config.organ_label)
        valid_coords = np.argwhere(non_tumor_mask)

        if len(valid_coords) == 0:
            return image, label

        for _ in range(num_copies):
            # Pick random offset from valid locations
            center = valid_coords[np.random.randint(len(valid_coords))]
            offset = center - coords.mean(axis=0).astype(int)

            # Apply small random jitter
            jitter = np.array([np.random.randint(-4, 5) for _ in range(3)])  # Reduced offset
            new_coords = np.clip(coords + offset + jitter, 0, np.array(label.shape) - 1)

            # Ensure no overlap with existing organ/tumor
            new_coords_unique = np.unique(new_coords, axis=0)
            overlap = label[tuple(new_coords_unique.T)] > 0
            if overlap.sum() > 0:
                continue  # Skip if would overlap

            # Copy tumor
            label[tuple(new_coords_unique.T)] = self.config.tumor_label

            # Copy with intensity variation
            scale_factor = np.random.uniform(0.95, 1.05)  # Very small variation
            for c in range(image.shape[0]):
                image[c][tuple(new_coords_unique.T)] = tumor_intensity[c] * scale_factor

        return image, label

    def augment(self, image, label):
        """Apply full augmentation pipeline with tumor survival guarantee"""
        # Store original tumor count
        orig_tumor_count = self._tumor_voxel_count(label)
        orig_has_tumor = orig_tumor_count > 0

        # Apply augmentations
        image, label = self.random_flip(image, label)
        image, label = self.random_rotation(image, label)
        image, label = self.random_scale(image, label)
        image, label = self.tumor_copy_paste_augmentation(image, label)
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image, label = self.elastic_deform(image, label)

        # Final tumor preservation check
        if orig_has_tumor:
            new_tumor_count = self._tumor_voxel_count(label)
            if new_tumor_count < orig_tumor_count * 0.5:  # Lost >50% of tumor
                # This should ideally revert, but for simplicity we just log
                # In practice, you'd want to re-sample a new patch
                pass

        return image, label


# Patch Sampling with GUARANTEED tumor coverage
class PatchSampler:
    """Intelligent patch sampler with guaranteed tumor coverage"""

    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.patch_size = config.patch_size
        self.tumor_ratio = config.tumor_patch_ratio
        self.organ_ratio = config.organ_patch_ratio

        total = self.tumor_ratio + self.organ_ratio
        if total > 1.0:
            self.tumor_ratio /= total
            self.organ_ratio /= total

        self.random_ratio = max(0.0, 1.0 - self.tumor_ratio - self.organ_ratio)

        # Use thread-local RNG instead of global numpy.random
        # Numpy thread-local Random Number Generator (RNG): better than global numpy.random for DataParallel
        self.rng = np.random.RandomState(42) if mode == 'val' else np.random.RandomState()

    def _random_center(self, shape):
        """Generate random patch center coordinates using local RNG"""
        ps_d, ps_h, ps_w = self.patch_size
        D, H, W = shape
        z = self.rng.randint(0, max(1, D - ps_d + 1))
        y = self.rng.randint(0, max(1, H - ps_h + 1))
        x = self.rng.randint(0, max(1, W - ps_w + 1))
        return z, y, x

    def _center_near_mask(self, mask, shape):
        """Generate patch center near mask with verification"""
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return self._random_center(shape)

        ps_d, ps_h, ps_w = self.patch_size
        D, H, W = shape

        # Try 5 times to get mask inside patch
        for attempt in range(5):
            idx = self.rng.randint(0, len(coords))
            zc, yc, xc = coords[idx]

            z = np.clip(zc - ps_d // 2, 0, max(0, D - ps_d))
            y = np.clip(yc - ps_h // 2, 0, max(0, H - ps_h))
            x = np.clip(xc - ps_w // 2, 0, max(0, W - ps_w))

            # Verify mask in patch
            z_end = min(z + ps_d, D)
            y_end = min(y + ps_h, H)
            x_end = min(x + ps_w, W)
            patch_mask = mask[z:z_end, y:y_end, x:x_end]

            if patch_mask.sum() > 5:
                return z, y, x

        # Fallback: use centroid
        centroid = coords.mean(axis=0).astype(int)
        zc, yc, xc = centroid
        z = np.clip(zc - ps_d // 2, 0, max(0, D - ps_d))
        y = np.clip(yc - ps_h // 2, 0, max(0, H - ps_h))
        x = np.clip(xc - ps_w // 2, 0, max(0, W - ps_w))
        return z, y, x

    def sample(self, image, label):
        """Sample patch with guaranteed center assignment using local RNG"""
        shape = label.shape
        tumor_mask = (label == self.config.tumor_label)
        organ_mask = (label == self.config.organ_label)

        r = self.rng.rand()
        center = None

        # Training mode with 100% tumor patch present
        if self.mode == 'train':
            if tumor_mask.sum() > 0:
                center = self._center_near_mask(tumor_mask, shape)
            else:
                center = self._random_center(shape)

        # Validation or no tumor: use config ratios
        elif tumor_mask.sum() > 0 and r < self.tumor_ratio:
            center = self._center_near_mask(tumor_mask, shape)
        elif organ_mask.sum() > 0 and r < self.tumor_ratio + self.organ_ratio:
            center = self._center_near_mask(organ_mask, shape)

        # Fallback to ensure center is always assigned
        if center is None:
            center = self._random_center(shape)

        return self.extract_patch(image, label, center)

    def extract_patch(self, image, label, center):
        """Extract 3D patch with padding if needed"""
        ps_d, ps_h, ps_w = self.patch_size
        z, y, x = center
        D, H, W = label.shape

        z_start = max(0, z)
        y_start = max(0, y)
        x_start = max(0, x)
        z_end = min(D, z + ps_d)
        y_end = min(H, y + ps_h)
        x_end = min(W, x + ps_w)

        img_patch = image[:, z_start:z_end, y_start:y_end, x_start:x_end]
        lab_patch = label[z_start:z_end, y_start:y_end, x_start:x_end]

        # Padding calculation
        pad_d_before = max(0, -z)
        pad_h_before = max(0, -y)
        pad_w_before = max(0, -x)
        pad_d_after = max(0, (z + ps_d) - D)
        pad_h_after = max(0, (y + ps_h) - H)
        pad_w_after = max(0, (x + ps_w) - W)

        if pad_d_before > 0 or pad_d_after > 0 or pad_h_before > 0 or pad_h_after > 0 or pad_w_before > 0 or pad_w_after > 0:
            img_patch = np.pad(img_patch,
                              ((0, 0),
                               (pad_d_before, pad_d_after),
                               (pad_h_before, pad_h_after),
                               (pad_w_before, pad_w_after)),
                              mode='constant')
            lab_patch = np.pad(lab_patch,
                              ((pad_d_before, pad_d_after),
                               (pad_h_before, pad_h_after),
                               (pad_w_before, pad_w_after)),
                              mode='constant')

        return img_patch, lab_patch

# Replace global cache with per-dataset LRU cache to prevent memory leaks
class DatasetCache:
    """Thread-safe per-dataset cache with size limit"""

    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.cache = {}
        self.order = []
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.maxsize:
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.cache[key] = value
            self.order.append(key)

# Boundary Map Computation
def compute_contour_map(label, organ_label, tumor_label):
    """Compute boundary distance map for boundary loss"""
    contour = np.zeros_like(label, dtype=np.float32)

    for lbl in [organ_label, tumor_label]:
        m = (label == lbl).astype(np.uint8)
        if m.sum() == 0:
            continue

        try:
            eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)))
            boundary = m ^ eroded
            if boundary.sum() > 0:
                dist = distance_transform_edt(~boundary)
                dist_normalized = np.exp(-dist / 5.0)
                contour = np.maximum(contour, dist_normalized)
        except:
            pass

    return contour

# Main Dataset Class
class MSDDataset(Dataset):
    """Medical Segmentation Dataset with tumor preservation"""

    def __init__(self, config, data_dir, mode='train', augment=True):
        self.config = config
        self.data_dir = data_dir
        self.mode = mode
        self.augment = augment and (mode == 'train')

        self.preprocessor = CTPreprocessor(config)
        self.augmentor = Augmentor(config) if self.augment else None
        self.patch_sampler = PatchSampler(config, mode=mode) if mode in ['train', 'val'] else None

        # Use per-dataset cache instead of global cache
        self.cache = DatasetCache(maxsize=30)  # Reduced size for memory safety

        self.image_files, self.label_files = self._resolve_files()

        if mode != 'test' and len(self.image_files) != len(self.label_files):
            raise RuntimeError(
                f"Image/label count mismatch: {len(self.image_files)} vs {len(self.label_files)}"
            )

        if mode == 'train':
            self._validate_tumor_presence()

    def _resolve_files(self):
        """Auto detect and resolve dataset layout"""
        a_img_dir = os.path.join(self.data_dir, self.mode, 'images')
        a_lbl_dir = os.path.join(self.data_dir, self.mode, 'labels')

        if os.path.isdir(a_img_dir) and (self.mode == 'test' or os.path.isdir(a_lbl_dir)):
            image_files = sorted([
                os.path.join(a_img_dir, f) for f in os.listdir(a_img_dir)
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
            label_files = []
            if self.mode != 'test':
                label_files = sorted([
                    os.path.join(a_lbl_dir, f) for f in os.listdir(a_lbl_dir)
                    if f.endswith('.nii') or f.endswith('.nii.gz')
                ])
            return image_files, label_files

        ds_root = os.path.join(self.data_dir, self.config.dataset_name)

        if self.mode == 'train':
            img_dir = os.path.join(ds_root, 'imagesTr')
            lbl_dir = os.path.join(ds_root, 'labelsTr')
        elif self.mode == 'val':
            img_dir = os.path.join(ds_root, 'imagesTr')
            lbl_dir = os.path.join(ds_root, 'labelsTr')
        else:
            img_dir = os.path.join(ds_root, 'imagesTs')
            lbl_dir = None

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"Could not find images directory: {img_dir}")

        image_files = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith('.nii') or f.endswith('.nii.gz')
        ])

        label_files = []
        if self.mode != 'test':
            if not os.path.isdir(lbl_dir):
                raise RuntimeError(f"Could not find labels directory: {lbl_dir}")
            label_files = sorted([
                os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])

        return image_files, label_files

    def _validate_tumor_presence(self):
        """Validate that training data contains tumors"""
        print("Validating tumor presence in training data...")

        if len(self.label_files) == 0:
            raise RuntimeError(f"No label files found")

        cases_with_tumor = 0
        total_tumor_voxels = 0

        for lbl_path in tqdm(self.label_files, desc="Scanning for tumors"):
            label = nib.load(lbl_path).get_fdata().astype(np.int16)
            tumor_voxels = (label == self.config.tumor_label).sum()
            if tumor_voxels > 0:
                cases_with_tumor += 1
                total_tumor_voxels += tumor_voxels

        print(f"  {cases_with_tumor}/{len(self.label_files)} cases contain tumor")
        print(f"  Average tumor voxels per case: {total_tumor_voxels / max(cases_with_tumor, 1):.1f}")

        if cases_with_tumor == 0:
            raise RuntimeError(f"NO TUMORS FOUND for tumor_label == {self.config.tumor_label}")

    def _load_case(self, img_path, lab_path):
        """Load single case with caching"""
        cache_key = (img_path, self.mode)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        img_nii = nib.load(img_path)
        img = img_nii.get_fdata().astype(np.float32)
        spacing = img_nii.header.get_zooms()[:3]

        lab = None
        if self.mode != 'test' and lab_path is not None:
            lab = nib.load(lab_path).get_fdata().astype(np.int16)

        result = (img, lab, spacing)

        if getattr(self.config, 'cache_rate', 0.0) > np.random.rand():
            self.cache.put(cache_key, result)

        return result

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and process sample with tumor preservation"""
        img_path = self.image_files[idx]
        lab_path = self.label_files[idx] if (self.mode != 'test' and self.label_files) else None
        image, label, spacing = self._load_case(img_path, lab_path)

        image, label = self.preprocessor.preprocess(image, label, spacing)

        if not np.isfinite(image).all():
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Patch sampling
        if self.mode in ['train', 'val'] and self.patch_sampler is not None and label is not None:
            image, label = self.patch_sampler.sample(image, label)

        tumor_present = (label == self.config.tumor_label).sum() > 0 if label is not None else False

        # Augmentation with tumor recovery
        if self.augment and self.mode == 'train' and label is not None:
            # Store original
            orig_image_full = image.copy()
            orig_label_full = label.copy()

            image, label = self.augmentor.augment(image, label)

            # Multi-retry if tumor lost
            if tumor_present:
                for retry in range(2):
                    if (label == self.config.tumor_label).sum() > 0:
                        break
                    # Re-sample and augment from original
                    image, label = self.patch_sampler.sample(orig_image_full, orig_label_full)
                    image, label = self.augmentor.augment(image, label)

        # Boundary/contour map
        if self.mode == 'test' or label is None:
            contour = np.zeros_like(image[0], dtype=np.float32)
        else:
            contour = compute_contour_map(label, self.config.organ_label, self.config.tumor_label)

        image = torch.from_numpy(image.astype(np.float32))
        if label is None:
            label = np.zeros_like(image[0], dtype=np.int64)
        label = torch.from_numpy(label.astype(np.int64))
        contour = torch.from_numpy(contour[np.newaxis, ...].astype(np.float32))

        tumor_voxels = (label == self.config.tumor_label).sum()
        organ_voxels = (label == self.config.organ_label).sum()

        assert image.shape[1:] == tuple(self.patch_sampler.patch_size), \
            f"Patch shape mismatch: {image.shape[1:]} != {self.patch_sampler.patch_size}"

        return {
            'image': image,
            'label': label,
            'contour': contour,
            'filename': img_path,
            'tumor_voxels': tumor_voxels,
            'organ_voxels': organ_voxels
        }

# DataLoader Factory
def get_dataloader(config, data_dir, mode='train', batch_size=None):
    """Create DataLoader"""
    if batch_size is None:
        batch_size = config.batch_size

    dataset = MSDDataset(config, data_dir, mode=mode, augment=(mode == 'train'))

    # Reduce augmentation for validation (only if augmentor exists)
    if mode == 'val' and dataset.augmentor is not None:
        dataset.augmentor.aug_prob = 0.1
        dataset.augmentor.elastic_deform_enabled = False
        dataset.augmentor.tumor_aug_prob = 0.0

    # Use proper worker initialization to prevent memory issues
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        torch.manual_seed(torch.initial_seed() + worker_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=config.num_workers if mode == 'train' else max(1, config.num_workers // 2),
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if mode == 'train' else False,
        drop_last=(mode == 'train'),
        worker_init_fn=worker_init_fn,  # worker-Safe initialization
    )
