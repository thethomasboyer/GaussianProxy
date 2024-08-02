from setuptools import find_packages, setup

# Read the contents of requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GaussianProxy",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
