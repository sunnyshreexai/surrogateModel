"""Setup script for SurrogateModel package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="surrogatemodel",
    version="1.0.0",
    author="Sunny Shree, Krishna Khadka, Yu Lei",
    author_email="sunny.shree@mavs.uta.edu",
    description="ML Model Construction Using Combinatorial Testing and Active Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunnyshreexai/surrogateModel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "optimization": [
            "optuna>=3.0.0",
        ],
        "deep": [
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "surrogatemodel=surrogatemodel.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "surrogatemodel": ["*.yaml", "*.json"],
    },
)