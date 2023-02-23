# PyDESeq2

## Table of Contents
- [PyDESeq2](#pydeseq2)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Getting started](#getting-started)
    - [Documentation](#documentation)
    - [Data](#data)
  - [Contributing](#contributing)
    - [1 - Download the repository](#1---download-the-repository)
    - [2 - Create a conda environment](#2---create-a-conda-environment)
  - [Development roadmap](#development-roadmap)
  - [Citing this work](#citing-this-work)
  - [References](#references)
  - [License](#license)

## Overview

This package is a python implementation of the [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) 
method [1] for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
It aims to facilitate DEA experiments for python users.

As PyDESeq2 is a re-implementation of [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) from 
scratch, you may experience some differences in terms of retrieved values or available features.

Currently, available features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor 
and paired multi-factor analysis (with bi-level factors), but we plan to implement more in the near future.
In case there is a feature you would particularly like to be implemented, feel free to open an issue.

## Installation

`PyDESeq2` can be installed from PyPI:

`pip install pydeseq2`

We recommend installing within a conda environment:

```
conda create -n pydeseq2
conda activate pydeseq2
pip install pydeseq2
```

If you're interested in contributing or want access to the development version, please see the [contributing](#contributing) section.

### Requirements

The list of package version requirements is available in `setup.py`.

For reference, the code is being tested in a github workflow (CI) with python
3.8-3.11 and the following package versions:
```
- numpy 1.23.0
- pandas 1.4.3
- scikit-learn 1.1.1
- scipy 1.8.1
- statsmodels 0.13.2
```

Please don't hesitate to open an issue in case you encounter any issue due to possible deprecations.


## Getting started

The [Getting Started](https://pydeseq2.readthedocs.io/en/latest/auto_examples/index.html) section of the documentation
contains downloadable examples on how to use PyDESeq2.


### Documentation

The documentation is hosted [here on ReadTheDocs](https://pydeseq2.readthedocs.io/en/latest/). 
If you want to have the latest version of the documentation, you can build it from source.
Please go to the dedicated [README.md](https://github.com/owkin/PyDESeq2/blob/main/docs/README.md) for information on how to do so.

### Data

The quick start examples use synthetic data, provided in this repo (see [datasets](https://github.com/owkin/PyDESeq2/blob/main/datasets/README.md).)

The experiments described in our [preprint](https://www.biorxiv.org/content/10.1101/2022.12.14.520412v1) rely on data
from [The Cancer Genome Atlas](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga),
which may be obtained from this [portal](https://portal.gdc.cancer.gov/).

## Contributing

Please the [Contributing](https://pydeseq2.readthedocs.io/en/latest/usage/contributing.html) section of the
documentation to see how you can contribute to PyDESeq2.

### 1 - Download the repository

`git clone https://github.com/owkin/PyDESeq2.git`

### 2 - Create a conda environment

Run `conda env create -n pydeseq2 python=3.8` (or higher python version) to create the `pydeseq2` environment and then activate it:
`conda activate pydeseq2`.

`cd` to the root of the repo and run `pip install -e ."[dev]"` to install in developer mode.

Then, run `pre-commit install`.

The `pre-commit` tool will automatically run [black](https://black.readthedocs.io/en/stable/)
and [isort](https://pycqa.github.io/isort/), and check [flake8](https://flake8.pycqa.org/en/latest/) compatibility

PyDESeq2 is a living project and any contributions are welcome! Feel free to open new PRs or issues.

## Development Roadmap

Here are some of the features and improvements we plan to implement in the future:

- [ ] Integration to the [scverse](https://scverse.org/) ecosystem:
  * [x] Refactoring to use the [AnnData](https://anndata.readthedocs.io/) data structure
  * [ ] Submitting a PR to be listed as an [scverse ecosystem](https://github.com/scverse/ecosystem-packages/) package
- [ ] Variance-stabilizing transformation
- [ ] Improving multi-factor analysis:
  * [ ] Allowing n-level factors (only bi-level for now)
  * [ ] Implementing interaction terms

## Citing this work

```
@article{muzellec2022pydeseq2,
  title={PyDESeq2: a python package for bulk RNA-seq differential expression analysis},
  author={Muzellec, Boris and Telenczuk, Maria and Cabeli, Vincent and Andreux, Mathieu},
  year={2022},
  doi = {10.1101/2022.12.14.520412},
  journal={bioRxiv},
}
```

## References

[1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>

[2] Zhu, A., Ibrahim, J. G., & Love, M. I. (2019).
        "Heavy-tailed prior distributions for sequence count data:
        removing the noise and preserving large differences."
        Bioinformatics, 35(12), 2084-2092.
        <https://academic.oup.com/bioinformatics/article/35/12/2084/5159452>

## License

PyDESeq2 is released under an [MIT license](https://github.com/owkin/PyDESeq2/blob/main/LICENSE).

