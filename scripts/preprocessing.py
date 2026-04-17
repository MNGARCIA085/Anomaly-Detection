import hydra
from omegaconf import DictConfig


from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)


if __name__ == "__main__":
    main()



# python -m scripts.preprocessing