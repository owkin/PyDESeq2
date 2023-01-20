"""
Getting started
===============

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
# obtained through PyDESeq2's API. You may replace it with your own dataset.

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
# In this example, the clinical data contains two columns, `condition` and `group`,
# representing two types of bi-level annotations. In the first part, we will only use the
# `condition` factor. Later on, we'll see how to use both the `condition` and the `group`
# factors in our analysis (see :ref:`multifactor_ref`).

# %%
# Data filtering
# ^^^^^^^^^^^^^^
#
# Before proceeding with DEA, it is good practice to preprocess your data, e.g. to remove
# samples for which annotations are missing and exclude genes with very low levels of
# expression. This is not necessary in the case of the synthetic data we provide, but we
# provide example code snippets to show how we could do this with real data.
#
# We start by removing samples for which "condition" is NaN. If you are using another
# dataset, do not forget to change "condition" for the column of `clinical_df` you wish
# to as a design factro in your analysis.

samples_to_keep = ~clinical_df.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
clinical_df = clinical_df.loc[samples_to_keep]

# %%
# .. note::
#  In the case where the design factor contains NaNs, PyDESeq2 will throw an error when
#  intializing a `DeseqDataSet`

# %%
# Next, we filter out genes that have less than 10 counts in total. Again, there are no
# such genes in this synthetic dataset.

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# %%
# Now that we have loaded and filtered our data, we may proceed with the differential
# analysis.


# %%
# Single factor analysis
# --------------------------

# %%
# Read counts modeling with the `DeseqDataSet` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We start by creating a `DeseqDataSet` object from the count and clinical data.
#
# Another option of interest is the `refit_cooks` argument
# (set to `True` by default), which controls whether Cooks outlier
# should be refitted – this is advised, in general.
#
# .. note::
#  in the case of the provided synthetic data, there won't be any Cooks
#  outliers.
#
# Here, we use 8 threads, feel free to adapt this to your setup or to set
# to `None` to use all available CPUs.
#
# Start by creating a DeseqDataSet
#
# .. note::
#   "condition" is a column in `clinical_df`. You might need to update it if you
#   use a different dataset.

dds = DeseqDataSet(
    counts_df,
    clinical_df,
    design_factors="condition",
    refit_cooks=True,
    n_cpus=8,
)

# %%
# Then, run DESeq2 on it
dds.deseq2()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "dds.pkl"), "wb") as f:
        pkl.dump(dds, f)

# %%
# Statistical analysis with the `DeseqStats` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `DeseqDataSet` class has a unique mandatory arguments, `dds`, which should
# be a *fitted* `DeseqDataSet` object, as well as a set of optional keyword
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
# The `summary` function runs the statistical analysis (multiple testing
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
# LFC shrinkage. This is implemented by the `lfc_shrink` method.

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
