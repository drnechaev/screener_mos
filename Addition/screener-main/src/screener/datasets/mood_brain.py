from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from multiprocessing import Pool
from tqdm.auto import tqdm
from functools import cached_property
import random
import nibabel
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.exposure import equalize_adapthist
from huggingface_hub import hf_hub_download, list_repo_files

from cotomka.datasets.base import Dataset
from cotomka.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation
from cotomka.utils.io import save_numpy, load_numpy

from .mood_abdomen import generate_anomaly_data


class MOODBrain(Dataset):
    name = 'screener/mood_brain'
    voxel_spacing = 0.7, 0.7875, 0.7

    @cached_property
    def ids(self) -> Tuple[str]:
        return tuple(sorted(file.name[:-len('.npy.gz')] for file in self.root_dir.glob('*.npy.gz')))

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / f'{id}.npy.gz', decompress=True).astype('float32')

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return self.voxel_spacing

    def prepare(self, num_workers: int = 1) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        ids = [
            filename[:-len('.nii.gz')]
            for filename in list_repo_files('dzimmerdkfz/mood_brain', repo_type='dataset')
            if filename.endswith('.nii.gz')
        ]

        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(self._prepare_image, ids), total=len(ids)))

    def _prepare_image(self, id: str) -> None:
        filepath = Path(hf_hub_download('dzimmerdkfz/mood_brain', repo_type='dataset', filename=f'{id}.nii.gz'))

        nii = nibabel.load(filepath)
        image = nii.get_fdata()
        affine = nii.affine

        voxel_spacing = affine_to_voxel_spacing(affine)
        image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)

        image = np.clip(image, 0.0, 1.0)
        image = equalize_adapthist(image, clip_limit=0.05)

        save_numpy(image.astype('float16'), self.root_dir / f'{id}.npy.gz', compression=1, timestamp=0)


class MOODBrainTrain(MOODBrain):
    @cached_property
    def ids(self) -> Tuple[str]:
        train_ids, test_ids = train_test_split(super().ids, test_size=10, random_state=42)
        return train_ids


class MOODBrainTest(MOODBrain):
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
