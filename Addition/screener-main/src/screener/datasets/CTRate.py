from typing import Tuple, Dict, Optional, List, Any
# from pathlib import Path
# from multiprocessing import Pool
# from tqdm.auto import tqdm
from functools import cached_property
import random
# import gzip
# import nibabel
# import math
import numpy as np
import pandas as pd
#from datasets import load_dataset
# from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from imops import zoom_to_shape

from cotomka.datasets.base import Dataset
# from cotomka.utils.io import save_numpy, load_numpy
from screener.utils import get_random_box



REPO_ID = '/home/jovyan/datasets/ct-rate/dataset/'


class _CTRATE(Dataset):
    _split: str

    @cached_property
    def labels_df(self):
        #print('getting labels')
        #print(pd.read_excel(self.root_dir / 'anatomy_segmentation_labels' / f'{self._split}_label_summary.xlsx').set_index('VolumeName'))
        return pd.read_excel(self.root_dir / 'anatomy_segmentation_labels' / f'{self._split}_label_summary.xlsx').set_index('VolumeName')

    @cached_property
    def metadata_df(self):
        return pd.read_csv(self.root_dir / 'metadata' / f'{self._split}_metadata.csv').set_index('VolumeName')

    @cached_property
    def reports_df(self):
        return pd.read_csv(self.root_dir / 'radiology_text_reports' / f'{self._split}_reports.csv').set_index('VolumeName')

    @cached_property
    def ids(self) -> Tuple[str]:
        return tuple(sorted(file.name[:-len('.nii.npz')] for file in (self.root_dir / f"{self._split}_fixed").glob('**/*.npz')))

    def _get_image(self, id: str) -> np.ndarray:
        folder_1, folder_2, folder_3, _ = id.split('_')
        folder_2 = folder_1 + '_' + folder_2
        folder_3 = folder_2 + '_' + folder_3
        subfolder = f'{folder_1}_fixed/{folder_2}/{folder_3}'
        return np.load(self.root_dir / subfolder / f'{id}.nii.npz')['img'].astype('float32')

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        
        return (1.0, 1.0, 1.0)
        #return (
        #    *map(float, self.metadata_df.loc[id, 'XYSpacing'][1:-1].split(', ')),
        #    float(self.metadata_df.loc[id, 'ZSpacing'])
        #)

    def _get_study_data(self, id: str) -> str:
        return self.metadata_df.loc[id, 'StudyDate']

    def _get_patient_sex(self, id: str) -> str:
        return self.metadata_df.loc[id, 'PatientSex']

    def _get_patient_age(self, id: str) -> str:
        return self.metadata_df.loc[id, 'PatientAge']

    def _get_technique(self, id: str) -> str:
        return self.reports_df.loc[id, 'Technique_EN']

    def _get_findings(self, id: str) -> str:
        return self.reports_df.loc[id, 'Findings_EN']

    def _get_impression(self, id: str) -> str:
        return self.reports_df.loc[id, 'Impressions_EN']

    def _get_labels(self, id: str) -> Dict[str, int]:
        return self.labels_df.loc[id].to_dict()

    def _get_patient_id(self, id: str) -> str:
        return '_'.join(id.split('_')[:2])

    def prepare(self, num_workers: int = 1) -> None:
        pass

    def _prepare_image(self, volume_name: str) -> None:
        pass
    
   


class CTRATETrain(_CTRATE):
    name = 'ct_rate'
    _split = 'train'
    
    # @cached_property
    # def ids(self) -> Tuple[str]:
    #     train_ids, test_ids = train_test_split(super().ids, test_size=10, random_state=42)
    #     return train_ids



class CTRATEVal(_CTRATE):
    name = 'ct_rate'
    _split = 'valid'
    
    @cached_property
    def ids(self) -> Tuple[str]:
        _, test_ids = train_test_split(super().ids, test_size=10, random_state=42)
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
