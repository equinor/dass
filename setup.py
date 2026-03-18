from setuptools import setup, find_packages

setup(
    name="dass",
    python_requires=">=3.13",
    packages=find_packages(include=["dass*"]),
    install_requires=[
        "p_tqdm>=1.3.3,<2.0.0",
        "matplotlib>=3.5.1,<4.0.0",
        "jupyter",
        "jupytext",
        "numpy",
        "pandas>=2.0.0",
        "scipy>=1.9.0",
    ],
)
