from typing import Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from monai.inferers import sliding_window_inference


class AnomalySegm(pl.LightningModule):
    def __init__(
            self,
            crop_size: Tuple[int, int, int] = (96, 96, 96),
            sw_batch_size: int = 4,
            lr: float = 3e-4,
            weight_decay: float = 1e-6
    ) -> None:
        super().__init__()

        self.crop_size = crop_size
        self.sw_batch_size = sw_batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def _per_crop_predictor(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(
            self,
            image: np.ndarray,
            voxel_spacing: Tuple[float, float, float],
            roi_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        anomaly_map = sliding_window_inference(
            inputs=image,
            roi_size=self.crop_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self._per_crop_predictor,
            overlap=0.5,
            mode='gaussian',
            sw_device=self.device,
            device='cpu'
        )
        anomaly_map = anomaly_map.squeeze(dim=(0, 1))
        anomaly_map = anomaly_map.data.numpy()
        if roi_mask is not None:
            anomaly_map = np.where(roi_mask, anomaly_map, anomaly_map.min())
        return {
            'anomaly_map': anomaly_map
        }

    def _on_eval_epoch_start(self) -> None:
        self._y_true = defaultdict(list)
        self._y_score = defaultdict(list)

    def _eval_step(self, batch: Dict) -> None:
        prediction = self.predict(batch['image'], batch['voxel_spacing'])
        anomaly_map = prediction['anomaly_map']

        pos_scores = anomaly_map[batch['anomaly_mask']]
        if len(pos_scores) > 1000:
            pos_scores = np.random.choice(pos_scores, 1000)
        self._y_score[batch['dataset_id']].extend(pos_scores)
        self._y_true[batch['dataset_id']].extend(np.ones(len(pos_scores), dtype=bool))

        neg_scores = anomaly_map[~batch['anomaly_mask']]
        if len(neg_scores) > 1000:
            neg_scores = np.random.choice(neg_scores, 1000)
        self._y_score[batch['dataset_id']].extend(neg_scores)
        self._y_true[batch['dataset_id']].extend(np.zeros(len(neg_scores), dtype=bool))

    def _on_eval_epoch_end(self):
        for k in self._y_true.keys():
            self.log(f'{k}/voxel_level_auroc', roc_auc_score(self._y_true[k], self._y_score[k]), on_epoch=True)

    def on_validation_epoch_start(self):
        return self._on_eval_epoch_start()

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        return self._eval_step(batch)

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end()

    def on_test_epoch_start(self):
        return self._on_eval_epoch_start()

    def test_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        return self._eval_step(batch)

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            eps=1e-3,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
