"""
Setup script for multimodal-tcell-classifier
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="multimodal-tcell-classifier",
    version="1.0.0",
    author="Polina Shirokikh",
    author_email="levchenkopg@icloud.com",
    description="Multimodal deep learning model for T-cell functional state prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polinavd/multimodal-tcell-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
    install_requires=requirements,
)
