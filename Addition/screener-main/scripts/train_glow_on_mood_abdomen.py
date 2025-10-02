from omegaconf import DictConfig
import hydra

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from screener.datasets.mood_abdomen import MOODAbdomenTrain, MOODAbdomenTest
from screener.lightning_datamodules.anomaly_segm import AnomalySegmDataModule
from screener.lightning_modules.glow import Glow


@hydra.main(version_base=None, config_path='../configs', config_name='train_glow_on_mood_abdomen')
def main(config: DictConfig):
    dm = AnomalySegmDataModule(
        train_datasets=[MOODAbdomenTrain()],
        test_datasets={'mood_abdomen_test': MOODAbdomenTest()},
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


if __name__ == '__main__':
    main()
