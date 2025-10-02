from typing import Tuple, Optional, Sequence, Union, List, NamedTuple
from omegaconf import DictConfig
import math
import time
import random
import numpy as np
from imops import crop_to_box, zoom
from scipy.ndimage import gaussian_filter1d, rotate

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from cotomka.datasets.base import Dataset as CotomkaDataset
from cotomka.utils.data_prefetcher import DataPrefetcher as CotomkaDataPrefetcher
from screener.data_prefetcher import DataPrefetcher
from screener.utils import get_random_box, normalize_axis_list


class ColorAugmentations(NamedTuple):
    blur_or_sharpen_p: float = 0.8
    blur_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_alpha_range: Tuple[float, float] = (0.0, 2.0)
    noise_p: float = 0.8
    noise_sigma_range: float = (0.0, 0.1)
    invert_p: float = 0.0
    brightness_p: float = 0.8
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_p: float = 0.8
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gamma_p: float = 0.8
    gamma_range: Tuple[float, float] = (0.8, 1.25)

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> 'ColorAugmentations':
        return cls(
            blur_or_sharpen_p=config.blur_or_sharpen_p,
            blur_sigma_range=tuple(config.blur_sigma_range),
            sharpen_sigma_range=tuple(config.sharpen_sigma_range),
            sharpen_alpha_range=tuple(config.sharpen_alpha_range),
            noise_p=config.noise_p,
            noise_sigma_range=tuple(config.noise_sigma_range),
            invert_p=config.invert_p,
            brightness_p=config.brightness_p,
            brightness_range=tuple(config.brightness_range),
            contrast_p=config.contrast_p,
            contrast_range=tuple(config.contrast_range),
            gamma_p=config.gamma_p,
            gamma_range=tuple(config.gamma_range)
        )


class SpatialAugmentations(NamedTuple):
    crop_size: Tuple[int, int, int] = (96, 96, 96)
    rot_p: float = 0.0
    max_angle: float = 30.0
    min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    flips: bool = False
    rot90: bool = False

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> 'SpatialAugmentations':
        return cls(
            crop_size=tuple(config.crop_size),
            rot_p=config.rot_p,
            max_angle=config.max_angle,
            min_voxel_spacing=tuple(config.min_voxel_spacing),
            max_voxel_spacing=tuple(config.max_voxel_spacing),
            flips=config.flips,
            rot90=config.rot90
        )


class Masking(NamedTuple):
    p: float = 0.0
    ratio: float = 0.6
    block_size: Tuple[int, int, int] = (24, 24, 24)

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> 'Masking':
        return cls(
            p=config.p,
            ratio=config.ratio,
            block_size=tuple(config.block_size)
        )


class DenseSSLDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_datasets: List[CotomkaDataset],
            spatial_augmentations: SpatialAugmentations = SpatialAugmentations(),
            color_augmentations: ColorAugmentations = ColorAugmentations(),
            masking: Masking = Masking(),
            num_voxels_per_crop: int = 1024,
            batch_size: int = 8,  # num images per batch
            num_batches_per_epoch: int = 1000,
    ) -> None:
        super().__init__()

        self._train_datasets = train_datasets
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.masking = masking
        self.num_voxels_per_crop = num_voxels_per_crop
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DataPrefetcher(
            dataset=_DenseSSLDataset(
                datasets=self._train_datasets,
                spatial_augmentations=self.spatial_augmentations,
                color_augmentations=self.color_augmentations,
                masking=self.masking,
                num_voxels_per_crop=self.num_voxels_per_crop,
                num_images_per_epoch=self.batch_size * self.num_batches_per_epoch,
            ),
            num_samples_per_epoch=self.batch_size * self.num_batches_per_epoch,
            num_workers=4,
            buffer_size=128,
            clone_factor=1,
            backend='threading'
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch: List) -> Tuple:
        return {
            'image_1': torch.from_numpy(np.stack([data['image_1'] for data in batch])),
            'mask_1': torch.from_numpy(np.stack([data['mask_1'] for data in batch])),
            'voxel_indices_1': [torch.from_numpy(data['voxel_indices_1']) for data in batch],
            'image_2': torch.from_numpy(np.stack([data['image_2'] for data in batch])),
            'mask_2': torch.from_numpy(np.stack([data['mask_2'] for data in batch])),
            'voxel_indices_2': [torch.from_numpy(data['voxel_indices_2']) for data in batch],
        }


class _DenseSSLDataset(Dataset):
    def __init__(
            self,
            datasets: List[CotomkaDataset],
            spatial_augmentations: SpatialAugmentations,
            color_augmentations: ColorAugmentations,
            masking: Masking,
            num_voxels_per_crop: int,
            num_images_per_epoch: int,
    ) -> None:
        super().__init__()

        self.data_prefetcher = CotomkaDataPrefetcher(
            *datasets,
            num_workers=4,
            buffer_size=128,
            clone_factor=16,
            backend='threading',
            fields=['image', 'voxel_spacing']
        )
        print('Waiting for 30 seconds for the data prefetcher to warm up...')
        time.sleep(30)

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.masking = masking
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch

    def __len__(self) -> int:
        return self.num_images_per_epoch

    def __getitem__(self, index: int) -> Tuple:
        data = next(self.data_prefetcher)
        data = _get_augmented_crops(
            image=data['image'],
            voxel_spacing=data['voxel_spacing'],
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            masking=self.masking,
            num_voxels_per_crop=self.num_voxels_per_crop,
        )
        return data


def _get_augmented_crops(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        spatial_augmentations: SpatialAugmentations,
        color_augmentations: ColorAugmentations,
        masking: Masking,
        num_voxels_per_crop: int,
) -> Tuple:
    for axis in range(3):
        if random.uniform(0, 1) < 0.5:
            image = np.flip(image, axis)

    rot90_k = random.randint(0, 3)
    image = np.rot90(image, k=rot90_k, axes=(0, 1))
    if rot90_k % 2:
        voxel_spacing = voxel_spacing[1], voxel_spacing[0], voxel_spacing[2]

    image_size = np.array(image.shape, dtype='int64')
    crop_size = np.array(spatial_augmentations.crop_size, dtype='int64')

    rot90_k_1 = random.choice([0, 1, 2, 3]) if spatial_augmentations.rot90 else 0
    rot90_k_2 = random.choice([0, 1, 2, 3]) if spatial_augmentations.rot90 else 0
    crop_size_before_rot90_1 = crop_size[[1, 0, 2]] if rot90_k_1 % 2 else crop_size
    crop_size_before_rot90_2 = crop_size[[1, 0, 2]] if rot90_k_2 % 2 else crop_size
    max_crop_size_before_rot90 = np.maximum(crop_size_before_rot90_1, crop_size_before_rot90_2)

    min_voxel_spacing = np.array(spatial_augmentations.min_voxel_spacing, dtype='float32')
    max_voxel_spacing = np.array(spatial_augmentations.max_voxel_spacing, dtype='float32')
    max_voxel_spacing = np.minimum(max_voxel_spacing, voxel_spacing * image_size / max_crop_size_before_rot90)
    voxel_spacing_1 = np.random.uniform(min_voxel_spacing, max_voxel_spacing)
    voxel_spacing_2 = np.random.uniform(min_voxel_spacing, max_voxel_spacing)
    crop_size_before_resize_1 = np.int64(np.round(crop_size_before_rot90_1 * voxel_spacing_1 / voxel_spacing))
    crop_size_before_resize_2 = np.int64(np.round(crop_size_before_rot90_2 * voxel_spacing_2 / voxel_spacing))
    resize_factor_1 = crop_size_before_rot90_1 / crop_size_before_resize_1
    resize_factor_2 = crop_size_before_rot90_2 / crop_size_before_resize_2

    min_crop_size_before_resize = np.minimum(crop_size_before_resize_1, crop_size_before_resize_2)
    overlap_size = np.round(np.random.uniform(0.25, 1.0, size=3) * min_crop_size_before_resize).astype('int16')
    union_size = np.minimum(image_size, crop_size_before_resize_1 + crop_size_before_resize_2 - overlap_size)
    overlap_size = crop_size_before_resize_1 + crop_size_before_resize_2 - union_size
    union_box = get_random_box(image_size, union_size)
    where_1 = np.random.uniform(0, 1, size=3) > 0.5
    where_2 = np.logical_not(where_1)
    crop_start_1 = np.where(where_1, union_box[0], union_box[1] - crop_size_before_resize_1)
    crop_start_2 = np.where(where_2, union_box[0], union_box[1] - crop_size_before_resize_2)
    crop_box_1 = np.array([crop_start_1, crop_start_1 + crop_size_before_resize_1])
    crop_box_2 = np.array([crop_start_2, crop_start_2 + crop_size_before_resize_2])
    overlap_start = np.maximum(crop_start_1, crop_start_2)
    overlap_box = np.array([overlap_start, overlap_start + overlap_size])

    voxel_indices = np.random.randint(overlap_box[0], overlap_box[1], size=(num_voxels_per_crop, 3))

    image_1, mask_1, voxel_indices_1 = _get_augmented_crop(
        image, voxel_spacing, voxel_indices, crop_box_1, resize_factor_1, rot90_k_1,
        spatial_augmentations, color_augmentations, masking
    )
    image_2, mask_2, voxel_indices_2 = _get_augmented_crop(
        image, voxel_spacing, voxel_indices, crop_box_2, resize_factor_2, rot90_k_2,
        spatial_augmentations, color_augmentations, masking
    )
    return {
        'image_1': image_1,
        'mask_1': mask_1,
        'voxel_indices_1': voxel_indices_1,
        'image_2': image_2,
        'mask_2': mask_2,
        'voxel_indices_2': voxel_indices_2
    }


def _get_augmented_crop(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        voxel_indices: np.ndarray,
        crop_box: np.ndarray,
        resize_factor: np.ndarray,
        rot90_k: int,
        spatial_augmentations: SpatialAugmentations,
        color_augmentations: ColorAugmentations,
        masking: Masking,
) -> Tuple:
    # crop
    image = crop_to_box(image, crop_box)
    voxel_indices = voxel_indices - crop_box[0]

    # random rotation
    if random.uniform(0, 1) < spatial_augmentations.rot_p:
        angle = np.random.uniform(-spatial_augmentations.max_angle, spatial_augmentations.max_angle)
        image = rotate(image, angle, axes=(-3, -2), reshape=False)
        voxel_indices = _rotate_voxel_indices(voxel_indices, angle, image.shape, image.shape)

    # resize
    image = zoom(np.ascontiguousarray(image), resize_factor, backend='Scipy')
    voxel_indices = np.int64(np.floor(voxel_indices * resize_factor))
    voxel_spacing = voxel_spacing / resize_factor

    # flips
    if spatial_augmentations.flips:
        for axis in [-3, -2, -1]:
            if random.uniform(0, 1) < 0.5:
                image = np.flip(image, axis)
                voxel_indices[:, axis] = image.shape[axis] - 1 - voxel_indices[:, axis]

    # rot90
    if rot90_k > 0:
        angle = 90 * rot90_k
        image_size_before_rot = image.shape
        image = np.rot90(image, k=rot90_k, axes=(0, 1))
        voxel_indices = _rotate_voxel_indices(voxel_indices, angle, image_size_before_rot, image.shape)

    image = image.copy()  # fix issues with flips

    # augment colors
    image = _augment_color(image, voxel_spacing, color_augmentations)

    # sample mask
    mask = _get_random_mask(image.shape, masking)

    # add channel dim
    image = np.expand_dims(image, axis=0)

    return image, mask, voxel_indices


def _rotate_voxel_indices(
        voxel_indices: np.ndarray,
        angle: float,
        image_size_before_rot: Tuple[int, int, int],
        image_size_after_rot: Tuple[int, int, int],
) -> np.ndarray:
    voxel_indices = voxel_indices.copy()
    image_size_before_rot = np.array(image_size_before_rot)
    image_size_after_rot = np.array(image_size_after_rot)

    voxel_ij_indices = voxel_indices[:, [0, 1]]
    voxel_ij_indices = voxel_ij_indices - (image_size_before_rot[[0, 1]] - 1) / 2.0
    angle = math.radians(angle)
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    voxel_ij_indices = voxel_ij_indices @ rot_matrix.T
    voxel_ij_indices = voxel_ij_indices + (image_size_after_rot[[0, 1]] - 1) / 2.0
    voxel_ij_indices = np.clip(voxel_ij_indices, 0, image_size_after_rot[[0, 1]] - 1)
    voxel_indices[:, [0, 1]] = voxel_ij_indices

    return voxel_indices


def _augment_color(
        image: np.ndarray,
        voxel_spacing: Sequence[float],
        color_augmentations: ColorAugmentations
) -> np.ndarray:
    voxel_spacing = np.array(voxel_spacing, dtype='float32')

    if random.uniform(0, 1) < color_augmentations.blur_or_sharpen_p:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(*color_augmentations.blur_sigma_range) / voxel_spacing[:2]
            image = _gaussian_filter(image, sigma, axis=(0, 1))
        else:
            sigma = random.uniform(*color_augmentations.sharpen_sigma_range) / voxel_spacing[:2]
            alpha = random.uniform(*color_augmentations.sharpen_alpha_range)
            image = _gaussian_sharpen(image, sigma, alpha, axis=(0, 1))

    if random.uniform(0, 1) < color_augmentations.noise_p:
        # gaussian noise
        noise_sigma = random.uniform(*color_augmentations.noise_sigma_range)
        image = image + np.random.normal(0, noise_sigma, size=image.shape).astype('float32')

    if random.uniform(0, 1) < color_augmentations.invert_p:
        # invert
        image = 1.0 - image

    if random.uniform(0, 1) < color_augmentations.brightness_p:
        # adjust brightness
        brightness_factor = random.uniform(*color_augmentations.brightness_range)
        image = np.clip(image * brightness_factor, image.min(), image.max())

    if random.uniform(0, 1) < color_augmentations.contrast_p:
        # adjust contrast
        contrast_factor = random.uniform(*color_augmentations.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast_factor + mean, image.min(), image.max())

    if random.uniform(0, 1) < color_augmentations.gamma_p:
        gamma = random.uniform(*color_augmentations.gamma_range)
        ptp = (image.max() - image.min())
        image = np.power((image - image.min()) / ptp, gamma) * ptp + image.min()

    return image


def _gaussian_filter(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    axis = normalize_axis_list(axis, x.ndim)
    sigma = np.broadcast_to(sigma, len(axis))
    for sgm, ax in zip(sigma, axis):
        x = gaussian_filter1d(x, sgm, ax)
    return x


def _gaussian_sharpen(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        alpha: float,
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    return x + alpha * (x - _gaussian_filter(x, sigma, axis))


def _get_random_mask(size: Sequence[int], masking: Masking) -> np.ndarray:
    if masking.ratio == 0.0 or random.uniform(0, 1) > masking.p:
        return np.ones(size, dtype='float32')

    size = np.array(size, dtype='int64')
    block_size = np.array(masking.block_size, dtype='int64')

    assert np.all(size % block_size == 0)

    mask = np.ones(size // block_size, dtype='float32')
    mask[np.unravel_index(np.random.permutation(mask.size)[:int(mask.size * masking.ratio)], mask.shape)] = 0.0
    assert (mask != 1.0).any()
    for axis, repeats in enumerate(block_size):
        mask = np.repeat(mask, repeats, axis)

    return mask
