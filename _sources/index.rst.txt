:github_url: https://github.com/IGNF/myria3d

Myria3D > Documentation
===================================================

.. include:: introduction.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/setup_install
   tutorials/prepare_dataset
   tutorials/make_predictions


.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/train_new_model
   guides/development

.. toctree::
   :maxdepth: 1
   :caption: Background

   background/interpolation
   background/data_optimization

.. TODO: assure that all dosctrings are in third-personn mode.
.. TODO: find a way to document hydra config ; perhaps by switching to a full dataclasses mode.

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   apidoc/scripts
   apidoc/configs
   apidoc/myria3d.data
   apidoc/myria3d.model
   apidoc/myria3d.models.modules
   apidoc/myria3d.callbacks
   apidoc/myria3d.utils


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`