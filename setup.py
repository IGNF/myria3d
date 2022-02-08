from setuptools import find_packages, setup

# Install from pip, here with no dependency b/c only pip dep are supported...
# In the future, prefer install as conda package,
# cf https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html

setup(
    name="lidar_multiclass",
    version="1.0.0",
    description="Multiclass Semantic Segmentation for Lidar Point Cloud",
    author="Charles GAYDON",
    author_email="",
    url="https://github.com/CharlesGaydon/Multiclass-Root-Model",  # replace with your own github project link
    install_requires=[
        # assume an environment as described in ./bash/setup_env.sh
    ],
    packages=find_packages(),
)
