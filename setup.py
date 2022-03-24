from setuptools import find_packages, setup

setup(
    name="lidar_multiclass",
    version="1.7.0",
    description="Multiclass Semantic Segmentation for Lidar Point Cloud",
    author="Charles GAYDON",
    url="https://github.com/CharlesGaydon/Multiclass-Root-Model",  # replace with your own github project link
    install_requires=[
        # assume an environment as described in ./bash/setup_env.sh
    ],
    packages=find_packages(),
)
