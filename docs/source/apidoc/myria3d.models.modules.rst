myria3d.models.modules
========================================

(Pytorch-Geometric) RandLA-Net
--------------------------------------------------

Our own implementation of RandLA-Net, to process full point clouds of different sizes, making the best use of the `pytorch-geometric framework <https://github.com/pyg-team/pytorch_geometric>`_.
Soon to be added to the framework `as an example <https://github.com/pyg-team/pytorch_geometric/pull/5117>`_.

.. autoclass:: myria3d.models.modules.pyg_randla_net.PyGRandLANet
   :members:

(Legacy) RandLA-Net
--------------------------------------------------

An open source implementation from `aRI0U <https://github.com/aRI0U/RandLA-Net-pytorch/>`_. Only accepts fixed-sized point clouds.

.. autoclass:: myria3d.models.modules.randla_net.RandLANet
   :members: