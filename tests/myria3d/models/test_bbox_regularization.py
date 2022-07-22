import pytest
import torch
from myria3d.models.model import get_bbox_regularization_weights

# importing numpy package
import numpy as np

# importing matplotlib package
import matplotlib.pyplot as plt
from matplotlib import cm

# importing mplot3d from
# mpl_toolkits
from mpl_toolkits import mplot3d


# @pytest.mark.parametrize("alpha", [0.25, 0.5, 1, 4, 8, 12, 16, 20])
@pytest.mark.parametrize("alpha", [1, 2, 3, 4, 5, 6])
def test_get_bbox_regularization_weights(alpha):
    x = torch.arange(-25, 25, 0.25)
    y = torch.arange(-25, 25, 0.25)
    x, y = np.meshgrid(x, y)
    x = torch.from_numpy(x.flatten())
    y = torch.from_numpy(y.flatten())
    pos = torch.stack([x, y]).permute(1, 0)
    batch = torch.cat(
        [torch.zeros((pos.size(0) // 2,)), torch.ones((pos.size(0) // 2,))]
    )
    width = torch.Tensor([10, 15])
    height = torch.Tensor([10, 15])
    cos_phi = torch.Tensor([1, 0])
    sin_phi = torch.Tensor([0, 1])
    bridge_mask = torch.ones_like(batch) == 1
    bbox_weight = get_bbox_regularization_weights(
        bridge_mask, pos, batch, width, height, cos_phi, sin_phi, alpha=alpha
    )
    idx = batch == 0
    pos = pos[idx].numpy()
    x = pos[:, 0]
    y = pos[:, 1]
    z = bbox_weight[idx].numpy()[:, 1]
    z = z

    # creating an empty canvas
    fig = plt.figure(figsize=(15, 15))

    # defining the axes with the projection
    # as 3D so as to plot 3D graphs
    ax = plt.axes(projection="3d")
    ax.set_zlim(-0.01, 1.01)
    # plotting a 3D line graph with X-coordinate,
    # Y-coordinate and Z-coordinate respectively
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    # plotting a scatter plot with X-coordinate,
    # Y-coordinate and Z-coordinate respectively
    # and defining the points color as cividis
    # and defining c as z which basically is a
    # defination of 2D array in which rows are RGB
    # or RGBA
    # Showing the above plot
    plt.show()
    fig.savefig(f"tests/logs/bbox_w_alpha_{alpha}.png", dpi=200)
