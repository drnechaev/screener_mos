from pathlib import Path
from omegaconf import DictConfig
import hydra

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from screener.datasets.CTRate import CTRATETrain
from screener.lightning_datamodules.dense_ssl import (
    DenseSSLDataModule, SpatialAugmentations, ColorAugmentations, Masking
)
from screener.lightning_modules.dense_vicreg import DenseVICReg


@hydra.main(version_base=None, config_path='../configs', config_name='train_dense_vicreg_on_ctrate')
def main(config: DictConfig):
    dm = DenseSSLDataModule(
        train_datasets=[CTRATETrain()],
        spatial_augmentations=SpatialAugmentations.from_dict_config(config.spatial_augmentations),
        color_augmentations=ColorAugmentations.from_dict_config(config.color_augmentations),
        masking=Masking.from_dict_config(config.masking),
        num_voxels_per_crop=config.num_voxels_per_crop,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches_per_epoch,
    )
    model = DenseVICReg(
        in_channels=1,
        descriptor_dim=config.descriptor_dim,
        i_weight=config.i_weight,
        v_weight=config.v_weight,
        c_weight=config.c_weight,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true',
        logger=TensorBoardLogger(
            save_dir=config.paths.output_dir,
            name=None,
            version='',
            log_graph=False
        ),
        max_steps=config.max_steps,
        profiler=SimpleProfiler(
            dirpath=config.paths.output_dir,
            filename='profile',
        ),
        gradient_clip_val=1.0
    )
    trainer.fit(model=model, datamodule=dm)
    torch.save(model.descriptor_model.state_dict(), '/home/jovyan/dumerenkov/__screener/screener-main/models/descriptor.pt')


if __name__ == '__main__':
    main()
