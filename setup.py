from setuptools import find_packages
from setuptools import setup

setup(
    name="pydeseq2",
    version="0.0.1",
    python_requires=">=3.8.0",
    license="",
    description="A python implementation of DESeq2.",
    author="Boris Muzellec, Maria Telenczuk, Vincent Cabelli and Mathieu Andreux",
    author_email="boris.muzellec@owkin.com",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "jupyter",
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.7.0",
        "statsmodels",
    ],  # external packages as dependencies
    extras_require={
        "dev": ["pytest>=6.2.4", "pre-commit>=2.13.0", "numpydoc"],
    },
)
