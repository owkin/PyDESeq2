.. PyDESeq2 documentation master file, created by
   sphinx-quickstart on Thu Dec 22 16:34:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyDESeq2's documentation!
====================================

This package is a python implementation of the [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) method [1]
for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
It aims to facilitate DEA experiments for python users.

As PyDESeq2 is a re-implementation of [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) from scratch, you may experience some differences in terms of retrieved values or available features.

Currently, available features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor analysis,
but we plan to implement more in the near future. In case there is a feature you would particularly like to be implemented, feel free to open an issue on GitHub.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   pydeseq2
   dataset

