import hydra
from omegaconf import OmegaConf

# A method used by hydra for partial instantiation of classes or functions
# see. https://github.com/facebookresearch/hydra/issues/1283
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method, replace=True)
