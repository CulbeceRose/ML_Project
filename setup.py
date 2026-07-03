from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() 
        and not line.startswith("#")
        and not line.startswith("-e")
    ]

setup(
    name = 'ML_Project',
    version = '0.0.1',
    author = 'RoseCulbece',
    packages = find_packages(),
    install_requires = requirements
)