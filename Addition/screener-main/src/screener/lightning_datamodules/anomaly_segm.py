from typing import Tuple, Optional, List, Any, Dict
import time
import numpy as np
import random
from imops import crop_to_box

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from cotomka.datasets.base import Dataset as CotomkaDataset
from cotomka.utils.data_prefetcher import DataPrefetcher as CotomkaDataPrefetcher

from screener.utils import get_random_box
from screener.data_prefetcher import DataPrefetcher


class AnomalySegmDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_datasets: List[CotomkaDataset],
            test_datasets: Dict[str, CotomkaDataset],
            crop_size: Tuple[int, int, int] = (96, 96, 96),
            batch_size: int = 8,
            num_batches_per_epoch: Optional[int] = 1000,
    ) -> None:
        super().__init__()

        self._train_datasets = train_datasets
        self._test_datasets = test_datasets
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DataPrefetcher(
            dataset=_TrainAnomalySegmDataset(
                datasets=self._train_datasets,
                crop_size=self.crop_size,
                num_images_per_epoch=self.batch_size * self.num_batches_per_epoch
            ),
            num_samples_per_epoch=self.batch_size * self.num_batches_per_epoch,
            num_workers=4,
            buffer_size=128,
            clone_factor=1,
            backend='threading'
        )
        self.test_datasets = [
            _TestAnomalySegmDataset(
                dataset=dataset,
                dataset_id=dataset_id
            )
            for dataset_id, dataset in self._test_datasets.items()
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader: 
        return [
            DataLoader(
                dataset,
                batch_size=None,
                collate_fn=lambda x: x,
                num_workers=0
            )
            for dataset in self.test_datasets
        ]

    def test_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                dataset,
                batch_size=None,
                collate_fn=lambda x: x,
                num_workers=0
            )
            for dataset in self.test_datasets
        ]

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if not self.trainer.training:
            # skip device transfer for the val and test dataloaders
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)


class _TrainAnomalySegmDataset(Dataset):
    def __init__(
            self,
            datasets: List[CotomkaDataset],
            crop_size: Tuple[int, int, int],
            num_images_per_epoch: int,
    ) -> None:
        super().__init__()

        self.data_prefetcher = CotomkaDataPrefetcher(
            *datasets,
            num_workers=4,
            buffer_size=128,
            clone_factor=16,
            backend='threading',
            fields=['image']
        )
        print('Waiting for 30 seconds for the data prefetcher to warm up...')
        time.sleep(30)

        self.crop_size = crop_size
        self.num_images_per_epoch = num_images_per_epoch

    def __len__(self) -> int:
        return self.num_images_per_epoch

    def __getitem__(self, index: int) -> Dict:
        image = next(self.data_prefetcher)['image']

        for axis in range(3):
            if random.uniform(0, 1) < 0.5:
                image = np.flip(image, axis)

        image = np.rot90(image, k=random.randint(0, 3), axes=(0, 1))

        image = image.copy()

        crop_box = get_random_box(image.shape, self.crop_size)
        image = crop_to_box(image, crop_box)

        image = np.expand_dims(image, axis=0)

        return {
            'image': image,
        }


class _TestAnomalySegmDataset(Dataset):
    def __init__(self, dataset: CotomkaDataset, dataset_id: str) -> None:
        super().__init__()

        self.dataset = dataset
        self.dataset_id = dataset_id

    def __len__(self) -> int:
        return len(self.dataset.ids)

    def __getitem__(self, index: int) -> Dict:
        id = self.dataset.ids[index]
        data = self.dataset.get(id, fields=['image', 'voxel_spacing', 'anomaly_mask'])
        data['dataset_id'] = self.dataset_id

        data['image'] = np.expand_dims(data['image'], axis=0)

        return data
