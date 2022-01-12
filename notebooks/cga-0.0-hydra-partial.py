# app.py
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.utils import instantiate

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@hydra.main(config_path=".", config_name="cga-0.0-hydra-partial.yaml")
def my_app(cfg: DictConfig) -> None:
    ooo = instantiate(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(ooo.opt2.opt)


my_app()
