from distutils.core import setup
from setuptools import find_packages


setup(
    name="dass",
    python_requires=">=3.8",
    packages=find_packages(include=["dass*"]),
    install_requires=[
        "p_tqdm>=1.3.3,<2.0.0",
        "matplotlib>=3.5.1,<4.0.0",
        "jupyter",
        "pandas>=1.4.2,<2.0.0",
        "scipy>=1.9.0,<2.0.0"
    ],
)
