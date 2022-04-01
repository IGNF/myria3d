from setuptools import find_packages, setup
import yaml

with open("package_metadata.yaml", "r") as f:
    pm = yaml.safe_load(f)

setup(
    name=pm["name"],
    version=pm["version"],
    url=pm.url,
    description=pm["description"],
    author=pm["author"],
    install_requires=[
        # assume an environment as described in ./bash/setup_env.sh
    ],
    packages=find_packages(),
)
