"""Packaging settings."""
import os
from codecs import open

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.md"), "r", "utf-8") as fp:
    readme = fp.read()

about: dict = dict()
with open(os.path.join(here, "pydeseq2", "__version__.py"), "r", "utf-8") as fp:
    exec(fp.read(), about)

setup(
    name="pydeseq2",
    version=about["__version__"],
    python_requires=">=3.8.0",
    license="MIT",
    description="A python implementation of DESeq2.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Boris Muzellec, Maria Telenczuk, Vincent Cabelli and Mathieu Andreux",
    author_email="boris.muzellec@owkin.com",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "anndata==0.8.0",
        "jupyter",
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.8.0",
        "statsmodels",
        "matplotlib>=3.6.2",  # not sure why sphinx_gallery does not work without it
    ],  # external packages as dependencies
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "pre-commit>=2.13.0",
            "numpydoc",
            "coverage",
            "mypy",
            "pandas-stubs",
        ],
    },
)
