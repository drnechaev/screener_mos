from omegaconf import DictConfig
import hydra

from screener.datasets.mood_brain import MOODBrain


@hydra.main(version_base=None, config_path='../configs', config_name='prepare_mood_brain')
def main(config: DictConfig):
    MOODBrain().prepare(num_workers=config.num_workers)


if __name__ == '__main__':
    main()
