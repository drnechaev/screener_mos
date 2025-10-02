from omegaconf import DictConfig
import hydra
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from screener.datasets.CTRate import CTRATETrain, CTRATEVal
from screener.lightning_datamodules.anomaly_segm import AnomalySegmDataModule
from screener.lightning_modules.glow import Glow


@hydra.main(version_base=None, config_path='../configs', config_name='train_glow_on_ctrate')
def main(config: DictConfig):
    dm = AnomalySegmDataModule(
        train_datasets=[CTRATETrain()],
        test_datasets={'ctrate_valid': CTRATEVal()},
        crop_size=config.crop_size,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches_per_epoch,
    )
    model = Glow(
        descriptor_model_path=config.descriptor_model_path,
        in_channels=1,
        descriptor_dim=config.descriptor_dim,
        avg_pool=config.avg_pool,
        avg_pool_kernel_size=config.avg_pool_kernel_size,
        avg_pool_stride=config.avg_pool_stride,
        avg_pool_padding=config.avg_pool_padding,
        sigma=config.sigma,
        glow_hidden_dim=config.glow_hidden_dim,
        glow_depth=config.glow_depth,
        crop_size=config.crop_size,
        sw_batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        precision='16-mixed',
        logger=TensorBoardLogger(
            save_dir='/home/jovyan/dumerenkov/__screener/glow_train2_',
            name=None,
            version='',
            log_graph=False
        ),
        max_steps=config.max_steps,
        profiler=SimpleProfiler(
            dirpath='/home/jovyan/dumerenkov/__screener/glow_train2_',
            filename='profile',
        ),
        strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=1.0
    )
    trainer.fit(model=model, datamodule=dm)
    torch.save(model, '/home/jovyan/dumerenkov/__screener/glow_train2_/glow_model.pt')


if __name__ == '__main__':
    main()
