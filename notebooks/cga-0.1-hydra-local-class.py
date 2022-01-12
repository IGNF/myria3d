# app.py
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.utils import instantiate

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


class foo:
    a = 5


@hydra.main(config_path=".", config_name="cga-0.1-hydra-local-class.yaml")
def my_app(cfg: DictConfig) -> None:
    ooo = instantiate(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(ooo.foo)


my_app()
