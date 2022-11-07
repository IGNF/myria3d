import numpy as np
import pytest

from myria3d.pctl.dataset.utils import get_mosaic_of_centers


@pytest.mark.parametrize(
    "tile_width, subtile_width, subtile_overlap",
    zip([1000], [50], [25]),
)
def test_get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap):
    mosaic = get_mosaic_of_centers(
        tile_width, subtile_width, subtile_overlap=subtile_overlap
    )
    for s in np.stack(mosaic).transpose():
        assert min(s - subtile_width / 2) <= 0
        assert max(s + subtile_width / 2) <= 1000
