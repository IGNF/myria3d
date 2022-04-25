import numpy as np
import pytest
from hydra.experimental import compose, initialize

# from lidar_prod.tasks.utils import pdal_read_las_array


@pytest.fixture
def default_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config")
