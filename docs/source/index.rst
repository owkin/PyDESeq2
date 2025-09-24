.. PyDESeq2 documentation master file, created by
   sphinx-quickstart on Thu Dec 22 16:34:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyDESeq2 documentation
======================

This package is a python implementation of the
`DESeq2 <https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_ method :cite:p:`love2014moderated`
for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
It aims to facilitate DEA experiments for python users.

As PyDESeq2 is a re-implementation of `DESeq2 <https://bioconductor.org/packages/release/bioc/html/DESeq2.html>`_
from scratch, you may experience some differences in terms of retrieved values or available features.

Currently, available features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor and
multi-factor analysis (with categorical or continuous factors) using Wald tests, with an
optional `apeGLM <https://academic.oup.com/bioinformatics/article/35/12/2084/5159452>`_ LFC shrinkage step
:cite:p:`zhu2019heavy`. We plan to implement more in the near future. In case there is a feature you would particularly
like to be implemented, feel free to open an issue on GitHub.


Citing this work
----------------

::

   @article{muzellec2023pydeseq2,
   title={PyDESeq2: a python package for bulk RNA-seq differential expression analysis},
   author={Muzellec, Boris and Telenczuk, Maria and Cabeli, Vincent and Andreux, Mathieu},
   year={2023},
   doi = {10.1093/bioinformatics/btad547},
   journal={Bioinformatics},
   }

License
-------

PyDESeq2 is released under an `MIT license <https://github.com/owkin/PyDESeq2/blob/main/LICENSE>`_.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: General

   usage/installation
   usage/requirements
   usage/contributing
   usage/references

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   auto_examples/index.rst
