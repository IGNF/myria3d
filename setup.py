from setuptools import find_packages, setup

setup(
    name="semantic_val",
    version="0.0.0",
    description="A 3D semantic segmentation model later used in validation of rules-based Lidar classification.",
    author="Charles GAYDON",
    author_email="charles.gaydon@gmail.com",
    # replace with your own github project link
    url="https://github.com/CharlesGaydon/Segmentation-Validation-Model",
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6"],
    packages=find_packages(),
)
