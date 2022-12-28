.. PyDESeq2 documentation master file, created by
   sphinx-quickstart on Thu Dec 22 16:34:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyDESeq2's documentation!
====================================

This package is a python implementation of the `DESeq2 <https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_ method [1]_
for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
It aims to facilitate DEA experiments for python users.

As PyDESeq2 is a re-implementation of `DESeq2 <https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_ from scratch, you may experience some differences in terms of retrieved values or available features.

Currently, available features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor analysis, with an optional `apeGLM <https://academic.oup.com/bioinformatics/article/35/12/2084/5159452>`_ LFC shrinkage [2]_.
We plan to implement more in the near future. In case there is a feature you would particularly like to be implemented, feel free to open an issue on GitHub.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
 
   api/index

Installation
------------

`PyDESeq2` can be installed from PyPI:

.. code-block:: bash 
   
   pip install pydeseq2

We recommend installing within a conda environment:

.. code-block:: python

   conda env create -n pydeseq2
   conda activate pydeseq2
   pip install pydeseq2


If you're interested in contributing or want access to the development version, please see the :ref: contributing section.

Requirements
------------

The list of package version requirements is available in  ``setup.py``, and will be automatically installed along PyDESeq2 when using PyPI.
For reference, the code was tested with python 3.8 and the following package versions:

- numpy 1.23.0
- pandas 1.4.3
- scikit-learn 1.1.1
- scipy 1.8.1
- statsmodels 0.13.2

Please don't hesitate to open an issue in case you encounter any issue due to possible deprecations.

Documentation
-------------

The documentation is automatically build using `Sphinx <https://www.sphinx-doc.org/en/master/>`_, and hosted `here on ReadTheDoc <https://pydeseq2.readthedocs.io/en/latest/>`_.

Getting started
---------------

The `notebooks directory <https://github.com/owkin/PyDESeq2/blob/main/notebooks/README.md>`_ contains minimal examples on how to use PyDESeq2, in the form of jupyter notebooks.

You can also try them from your browser (on synthetic data only): |Binder|

.. |Binder| image:: https://mybinder.org/badge_logo.svg 
   :target: https://mybinder.org/v2/gh/owkin/PyDESeq2/HEAD?labpath=notebooks%2Findex.ipynb

Contributing
------------

PyDESeq2 is a living project and any contributions are welcome! The project is hosted on `Github <https://github.com/owkin/PyDESeq2>`_ .
Feel free to come and open new PRs or issues.

Citing this work
----------------

::  

   @article{muzellec2022pydeseq2,
   title={PyDESeq2: a python package for bulk RNA-seq differential expression analysis},
   author={Muzellec, Boris and Telenczuk, Maria and Cabeli, Vincent and Andreux, Mathieu},
   year={2022},
   doi = {10.1101/2022.12.14.520412},
   journal={bioRxiv},
   }

References
----------

.. [1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>

..  [2] Zhu, A., Ibrahim, J. G., & Love, M. I. (2019).
        "Heavy-tailed prior distributions for sequence count data:
        removing the noise and preserving large differences."
        Bioinformatics, 35(12), 2084-2092.
        <https://academic.oup.com/bioinformatics/article/35/12/2084/5159452>
        
License
-------

PyDESeq2 is released under an `MIT license <https://github.com/owkin/PyDESeq2/blob/main/LICENSE>`_.





