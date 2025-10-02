from omegaconf import DictConfig
import hydra

from screener.datasets.mood_abdomen import MOODAbdomen


@hydra.main(version_base=None, config_path='../configs', config_name='prepare_mood_abdomen')
def main(config: DictConfig):
    MOODAbdomen().prepare(
        voxel_spacing=tuple(config.voxel_spacing),
        min_image_size=tuple(config.min_image_size),
        num_workers=config.num_workers
    )


if __name__ == '__main__':
    main()
