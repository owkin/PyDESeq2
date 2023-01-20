"""
Getting started
===============

.. currentmodule:: pydeseq2

In this example, we show how to perform a simple differential expression analysis on bulk
RNAseq data, using PyDESeq2.

.. contents:: Contents
    :local:
    :depth: 3

We start by importing required packages and setting up an optional path to save results.
"""

import os
import pickle as pkl

from pydeseq2.DeseqDataSet import DeseqDataSet
from pydeseq2.DeseqStats import DeseqStats
from pydeseq2.utils import load_example_data

SAVE = False  # whether to save the outputs of this notebook

if SAVE:
    # Replace this with the path to directory where you would like results to be saved
    OUTPUT_PATH = "../output_files/synthetic_example"
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# %%
# Data loading
# ------------
#
# To perform differential expression analysis (DEA), PyDESeq2 requires two types of
# inputs:
#
#   * A count matrix of shape #samples x #genes, containing read counts
#     (non-negative integers),
#   * Clinical data (or "column" data), containing sample annotations that will be used
#     to split the data in cohorts.
#
# Both should be provided as `pandas dataframes
# <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
#
# To illustrate the required data format, we load a synthetic example dataset that may be
# obtained through PyDESeq2's API using :func:`utils.load_example_data`.
# You may replace it with your own dataset.

counts_df = load_example_data(
    modality="raw_counts",
    dataset="synthetic",
    debug=False,
)

clinical_df = load_example_data(
    modality="clinical",
    dataset="synthetic",
    debug=False,
)

print(counts_df)

# %%
print(clinical_df)


# %%
# In this example, the clinical data contains two columns, ``condition`` and ``group``,
# representing two types of bi-level annotations. In the first part, we will only use the
# ``condition`` factor. Later on, we'll see how to use both the `condition` and the
# ``group`` factors in our analysis (see :ref:`multifactor_ref`).

# %%
# Data filtering
# ^^^^^^^^^^^^^^
#
# Before proceeding with DEA, it is good practice to preprocess your data, e.g. to remove
# samples for which annotations are missing and exclude genes with very low levels of
# expression. This is not necessary in the case of the synthetic data we provide, but we
# provide example code snippets to show how we could do this with real data.
#
# We start by removing samples for which ``condition`` is ``NaN``. If you are using
# another dataset, do not forget to change "condition" for the column of ``clinical_df``
# you wish to as a design factor in your analysis.

samples_to_keep = ~clinical_df.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
clinical_df = clinical_df.loc[samples_to_keep]

# %%
# .. note::
#   In the case where the design factor contains ``NaN`` entries, PyDESeq2 will throw an
#   error when intializing a :class:`DeseqDataSet <DeseqDataSet.DeseqDataSet>`.

# %%
# Next, we filter out genes that have less than 10 read counts in total. Note again that
# there are no such genes in this synthetic dataset.

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# %%
# Now that we have loaded and filtered our data, we may proceed with the differential
# analysis.


# %%
# Single factor analysis
# --------------------------
#
# In this first analysis, we ignore the ``group`` variable and use the ``condition``
# column as our design factor. That is, we compare gene expressions of samples that have
# ``condition B`` to those that have ``condition A``.
#

# %%
# .. currentmodule:: pydeseq2.DeseqDataSet
#
# Read counts modeling with the :class:`DeseqDataSet` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We start by creating a :class:`DeseqDataSet`
# object from the count and clinical data.
# A :class:`DeseqDataSet` fits dispersion and
# log-fold change (LFC) parameters from the data, and stores them.
#

dds = DeseqDataSet(
    counts_df,
    clinical_df,
    design_factors="condition",
    refit_cooks=True,
    n_cpus=8,
)

# %%
# A :class:`DeseqDataSet` has two mandatory
# arguments: a ``counts`` and a ``clinical`` dataframe, like the ones we've loaded in the
# first part of this tutorial.
#
# Next, we should specify the ``design_factor``, i.e. the column of the ``clinical``
# dataframe that will be used to compare samples. This can be a single string as above,
# or a list of strings, as in the
# :ref:`section on multifactor analysis<multifactor_ref>`.
#
# .. note::
#   ``"condition"`` is a column from the ``clinical_df`` dataframe we loaded earlier.
#   You might need to change it according to your own dataset.
#
# Several other arguments may be optionally specified (see the :doc:`API documentation
# </api/docstrings/pydeseq2.DeseqDataSet.DeseqDataSet>`).
# Among those, the ``refit_cooks`` argument (set to ``True`` by default), controls
# whether Cooks outlier should be refitted (which is advised, in general) and ``n_cpus``
# sets the number of CPUs to use for computation. Here, we use 8 threads. Feel free to
# adapt this to your setup or to set to ``None`` to use all available CPUs.
#
# .. note::
#     In the case of the provided synthetic data, there won't be any Cooks outliers.
#
# Once a :class:`DeseqDataSet` was initialized,
# we may run the :meth:`deseq2() <DeseqDataSet.deseq2>` method
# to fit dispersions and LFCs.
#
#


dds.deseq2()


if SAVE:
    with open(os.path.join(OUTPUT_PATH, "dds.pkl"), "wb") as f:
        pkl.dump(dds, f)

# %%
# If needed, we may now access the fitted dispersions and LFCs (in natural log scale):

print(dds.dispersions)

# %%

print(dds.LFCs)

# %%
# Statistical analysis with the ``DeseqStats`` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``DeseqStats`` class has a unique mandatory arguments, ``dds``, which should
# be a *fitted* :py:class:`DeseqDataSet <pydeseq2.DeseqDataSet.DeseqDataSet>`
# object, as well as a set of optional keyword
# arguments, among which:
#
# - `alpha`: the p-value and adjusted p-value significance threshold (0.05 by default),
# - `cooks_filter`: whether to filter p-values based on cooks outliers (True by default),
# - `independent_filter`: whether to perform independent filtering to correct
#   p-value trends (True by default).
#

stat_res = DeseqStats(dds, n_cpus=8)

# %%
# Wald test
# """"""""""
#
# The ``summary()`` method runs the statistical analysis (multiple testing
# adjustement included) and returns a summary DataFrame.

stat_res.summary()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "stat_results.pkl"), "wb") as f:
        pkl.dump(stat_res, f)

# %%
# LFC shrinkage
# """""""""""""
#
# For visualization or post-processing purposes, it might be suitable to perform
# LFC shrinkage. This is implemented by the ``lfc_shrink()`` method.

stat_res.lfc_shrink()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "shrunk_stat_results.pkl"), "wb") as f:
        pkl.dump(stat_res, f)

# %%
# .. _multifactor_ref:
#
# Multifactor analysis
# ---------------------
#
