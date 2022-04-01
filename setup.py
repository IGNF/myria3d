from setuptools import find_packages, setup
import yaml

with open("package_metadata.yaml", "r") as f:
    pm = yaml.safe_load(f)

setup(
    name=pm["__name__"],
    version=pm["__version__"],
    url=pm["__url__"],
    description=pm["__description__"],
    author=pm["__author__"],
    install_requires=[
        # assume an environment as described in ./bash/setup_env.sh
    ],
    packages=find_packages(),
)
