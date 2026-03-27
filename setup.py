"""
Setup script for tcell-classifier.

Install:
    pip install .                    # from cloned repo
    pip install tcell-classifier     # from PyPI (when published)

Usage after install:
    tcell-predict input.h5ad -o results/
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tcell-classifier",
    version="2.0.0",
    author="Polina Shirokikh",
    author_email="levchenkopg@icloud.com",
    description="Predict T-cell functional states from scRNA-seq + TCR data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polinavd/multimodal-tcell-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "tcell-predict=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scanpy>=1.9.0",
        "anndata>=0.9.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scirpy>=0.13.0",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "train": ["pyyaml>=6.0", "tqdm>=4.65.0"],
    },
)
