from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from multiprocessing import Pool
from tqdm.auto import tqdm
from functools import cached_property, partial
import nibabel
import numpy as np
import random
from sklearn.model_selection import train_test_split
from skimage.exposure import equalize_adapthist
from imops import crop_to_box, zoom, zoom_to_shape
from huggingface_hub import hf_hub_download, list_repo_files

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation
from cotomka.preprocessing.common import mask_to_bbox
from cotomka.utils.io import save_json, save_numpy, load_numpy, load_json

from screener.utils import get_random_box


class MOODAbdomen(Dataset):
    name = 'screener/mood_abdomen'

    @cached_property
    def ids(self) -> Tuple[str]:
        return tuple(sorted(file.name[:-len('.npy.gz')] for file in self.root_dir.glob('*.npy.gz')))

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / f'{id}.npy.gz', decompress=True).astype('float32')

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return load_json(self.root_dir / 'voxel_spacing.json')

    def prepare(
            self,
            voxel_spacing: Tuple[float, float, float],
            min_image_size: Tuple[int, int, int],
            num_workers: int = 1
    ) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        save_json(voxel_spacing, self.root_dir / 'voxel_spacing.json')

        ids = [
            filename[:-len('.nii.gz')]
            for filename in list_repo_files('dzimmerdkfz/mood_abdomen', repo_type='dataset')
            if filename.endswith('.nii.gz')
        ]
        prepare_image = partial(self._prepare_image, voxel_spacing=voxel_spacing, min_image_size=min_image_size)

        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(prepare_image, ids), total=len(ids)))

    def _prepare_image(
            self,
            id: str,
            voxel_spacing: Tuple[float, float, float],
            min_image_size: Tuple[int, int, int],
    ) -> None:
        filepath = Path(hf_hub_download('dzimmerdkfz/mood_abdomen', repo_type='dataset', filename=f'{id}.nii.gz'))

        try:
            nii = nibabel.load(filepath)
            image = nii.get_fdata()
            affine = nii.affine
        except Exception:
            return

        original_voxel_spacing = affine_to_voxel_spacing(affine)
        image, original_voxel_spacing = to_canonical_orientation(image, original_voxel_spacing, affine)

        # zoom to config.voxel_spacing
        image = image.astype('float32')
        scale_factor = tuple(original_voxel_spacing[i] / voxel_spacing[i] for i in range(3))
        image = zoom(image, scale_factor, fill_value=np.min, backend='Scipy')

        # zoom may pad image with zeros
        box = mask_to_bbox(image > image.min())
        image = crop_to_box(image, box, num_threads=-1, backend='Scipy')

        if any(image.shape[i] < min_image_size[i] for i in range(3)):
            return

        # rescale
        image = (image - image.min()) / (image.max() - image.min())
        image = equalize_adapthist(image, clip_limit=0.05)

        save_numpy(image.astype('float16'), self.root_dir / f'{id}.npy.gz', compression=1, timestamp=0)


class MOODAbdomenTrain(MOODAbdomen):
    @cached_property
    def ids(self) -> Tuple[str]:
        train_ids, test_ids = train_test_split(super().ids, test_size=10, random_state=42)
        return train_ids


class MOODAbdomenTest(MOODAbdomen):
    @cached_property
    def ids(self) -> Tuple[str]:
        train_ids, test_ids = train_test_split(super().ids, test_size=10, random_state=42)
        return test_ids

    @cached_property
    def fields(self) -> Tuple[str]:
        return ('image', 'voxel_spacing', 'anomaly_mask')

    def get(self, id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.root_dir.exists():
            raise OSError('Dataset is not prepared on this OS')

        if fields is None:
            fields = self.fields

        missing_fields = list(set(fields) - set(self.fields))
        if missing_fields:
            raise ValueError(f'Dataset does not contain fields {missing_fields}')

        data = dict()
        if 'image' in fields or 'anomaly_mask' in fields:
            image = self._get_image(id)
            other_id = random.choice(list(set(self.ids) - {id}))
            other_image = self._get_image(other_id)
            image, anomaly_mask = generate_anomaly_data(image, other_image)
            if 'image' in fields:
                data['image'] = image
            if 'anomaly_mask' in fields:
                data['anomaly_mask'] = anomaly_mask
        if 'voxel_spacing' in fields:
            data['voxel_spacing'] = self._get_voxel_spacing(id)
        data['id'] = id
        return data


BLOB_MASK_SIZE = 64, 64, 64


def generate_anomaly_data(image: np.ndarray, other_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image = image.copy()
    anomaly_mask = np.zeros(image.shape, dtype='bool')
    num_blobs = np.random.poisson(lam=5.0)
    for _ in range(num_blobs):
        blob_mask = _generate_blob_mask()
        start, stop = get_random_box(image.shape, BLOB_MASK_SIZE)
        other_start, other_stop = get_random_box(other_image.shape, BLOB_MASK_SIZE)
        
        image[tuple(map(slice, start, stop))] = np.where(
            blob_mask,
            other_image[tuple(map(slice, other_start, other_stop))],
            image[tuple(map(slice, start, stop))]
        )
        anomaly_mask[tuple(map(slice, start, stop))] = blob_mask

    return image, anomaly_mask


def _generate_blob_mask() -> np.ndarray:
    mask_size = np.array(BLOB_MASK_SIZE, dtype='int16')
    sigma = random.randint(2, 16)
    noise_size = mask_size // random.randint(12, 16)
    noise = np.random.uniform(0, 1, size=noise_size)
    noise = zoom_to_shape(noise, shape=mask_size, order=3, backend='Scipy')
    meshgrid = np.stack(np.meshgrid(*map(np.arange, mask_size), indexing='ij'), axis=-1)
    dist_from_center = np.linalg.norm(meshgrid + 0.5 - mask_size / 2, axis=-1)
    noise *= np.exp(-dist_from_center ** 2 / (2 * sigma ** 2))
    return noise >= 0.25
