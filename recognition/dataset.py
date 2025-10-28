import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class HipMRIDataset(Dataset):
    """Dataset class for loading HipMRI 3D volumes with optional downsampling and patch extraction.

    The dataset loads a 3D volume and its corresponding segmentation mask from disk
    using nibabel. To accelerate training on large volumes, optional downsampling
    and random 3D patch extraction can be enabled. Downsampling is performed by
    simple strides (e.g. taking every second voxel) to avoid heavy interpolation
    dependencies. Patches are sampled uniformly within the volume boundaries.

    Args:
        image_paths (List[str]): Paths to the 3D MRI volumes.
        label_paths (List[str]): Paths to the corresponding segmentation masks.
        downsample_factor (int): Step size for strided downsampling. A value of 1
            means no downsampling, 2 halves each spatial dimension, etc.
        patch_size (Tuple[int, int, int] or None): If provided, a random 3D patch
            of this size is cropped from each volume. If None, the entire volume
            (possibly downsampled) is returned.
    """

    def __init__(self, image_paths, label_paths, downsample_factor: int = 1, patch_size: tuple = None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.downsample_factor = max(1, downsample_factor)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load full 3D volumes from NIfTI files
        img_vol = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        mask_vol = nib.load(self.label_paths[idx]).get_fdata().astype(np.float32)

        # Optional downsampling via simple striding; preserves segmentation boundaries
        if self.downsample_factor > 1:
            f = self.downsample_factor
            img_vol = img_vol[::f, ::f, ::f]
            mask_vol = mask_vol[::f, ::f, ::f]

        # Optional random 3D patch cropping
        if self.patch_size is not None:
            d, h, w = mask_vol.shape
            pd, ph, pw = self.patch_size
            # Ensure patch fits within the volume
            pd = min(pd, d)
            ph = min(ph, h)
            pw = min(pw, w)
            start_d = 0 if d == pd else np.random.randint(0, d - pd + 1)
            start_h = 0 if h == ph else np.random.randint(0, h - ph + 1)
            start_w = 0 if w == pw else np.random.randint(0, w - pw + 1)
            img_vol = img_vol[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
            mask_vol = mask_vol[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]

        # Normalize the image volume to zero mean and unit variance to stabilize training
        img_vol = (img_vol - img_vol.mean()) / (img_vol.std() + 1e-8)
        # Expand channel dimension and binarize the mask
        img = np.expand_dims(img_vol, axis=0)
        mask = np.expand_dims((mask_vol > 0).astype(np.float32), axis=0)
        return {
            'image': torch.tensor(img, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
        }

