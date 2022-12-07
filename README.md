# PyDESeq2

This package is a python implementation of the [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) method [1]
for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
It aims to facilitate DEA experiments for python users.

Currently, available features broadly correspond to the default setting of
DESeq2 (TODO: version?), but we plan to implement more in the
near future. In case there is a feature you would particularly like to be
implemented, feel free to open an issue or contribute by opening a PR.

# Installation

## Create a conda environment

`conda create -n pydeseq2 python=3.8`, and then activate it:
`conda activate pydeseq2`

Run `pip install -e .` to install.


# Getting started

The `notebooks` directory contains minimal examples on how to use PyDESeq2 in the form of jupyter notebooks.

# Contributing

Run `pip install -e ."[dev]"` to install in developer mode.

Then, run `pre-commit install`.

The `pre-commit` tool will automatically run [black](https://black.readthedocs.io/en/stable/)
and [isort](https://pycqa.github.io/isort/), and check [flake8](https://flake8.pycqa.org/en/latest/) compatibility

PyDESeq2 is a living project and any contributions are welcome, in the form of PRs or new issues.

# Requirements

The list of package version requirements is available in `setup.py`.

For reference, the code was tested with the following package versions:
```
- numpy 1.23.0
- pandas 1.4.3
- scikit-learn 1.1.1
- scipy 1.8.1
```

Please don't hesitate to open an issue in case you encounter any issue due to possible deprecations.

### TCGA Data

See [datasets](./datasets/README.md).

# Citing this work

TBD

# References

[1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>
