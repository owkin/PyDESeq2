"""
A simple PyDESeq2 workflow
===========================

In this example, we show how to perform a simple differential expression analysis on bulk
RNAseq data, using PyDESeq2.

.. contents:: Contents
    :local:
    :depth: 3

We start by importing required packages and setting up an optional path to save results.

"""

import os
import pickle as pkl

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
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
#   * A count matrix of shape 'number of samples' x 'number of genes', containing
#     read counts (non-negative integers),
#   * Clinical data (or "column" data) of shape 'number of samples' x
#     'number of variables', containing sample annotations that will be used
#     to split the data in cohorts.
#
# Both should be provided as `pandas dataframes
# <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
#
# .. currentmodule:: pydeseq2
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
# expression. This is not necessary in the case of our synthetic data, but don't forget
# this step if you are using real data. To this end you can use the code below.
#
# We start by removing samples for which ``condition`` is ``NaN``. If you are using
# another dataset, do not forget to change "condition" for the column of ``clinical_df``
# you wish to use as a design factor in your analysis.

samples_to_keep = ~clinical_df.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
clinical_df = clinical_df.loc[samples_to_keep]

# %%
# .. note::
#   In the case where the design factor contains ``NaN`` entries, PyDESeq2 will throw an
#   error when intializing a :class:`DeseqDataSet <dds.DeseqDataSet>`.

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
# .. currentmodule:: pydeseq2.dds
#
# Read counts modeling with the :class:`DeseqDataSet` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We start by creating a :class:`DeseqDataSet`
# object from the count and clinical data.
# A :class:`DeseqDataSet` fits dispersion and
# log-fold change (LFC) parameters from the data, and stores them.
#

dds = DeseqDataSet(
    counts=counts_df,
    clinical=clinical_df,
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
#   The ``"condition"`` argument passed to ``design_factors`` corresponds to a column
#   from the ``clinical_df`` dataframe we loaded earlier.
#   You might need to change it according to your own dataset.
#
# Several other arguments may be optionally specified (see the :doc:`API documentation
# </api/docstrings/pydeseq2.dds.DeseqDataSet>`).
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
# The :class:`DeseqDataSet` class extends the
# :class:`AnnData <anndata.AnnData>`
# class.

print(dds)

# %%
# Hence, parameters are stored according to the :class:`AnnData <anndata.AnnData>` data
# structure, with key-based data fields. In particular,
#
# - ``X`` stores the count data,
# - ``obs`` stores design factors,
# - ``obsm`` stores sample-level data, such as ``"design_matrix"`` and
#   ``"size_factors"``,
# - ``varm`` stores gene-level data, such as ``"dispersions"`` and ``"LFC"``.
#
#
# As an example, here is how we would access dispersions and LFCs
# (in natural log scale):

# %%

print(dds.varm["dispersions"])

# %%

print(dds.varm["LFC"])

# %%
# .. currentmodule:: pydeseq2.ds
#
# Statistical analysis with the :class:`DeseqStats` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that dispersions and LFCs were fitted, we may proceed with statistical tests to
# compute p-values and adjusted p-values for differential expresion. This is the role of
# the :class:`DeseqStats` class. It has a unique mandatory argument, ``dds``, which
# should be a *fitted* :class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>`
# object.

stat_res = DeseqStats(dds, n_cpus=8)

# %%
# It also has a set of optional keyword arguments (see the :doc:`API documentation
# </api/docstrings/pydeseq2.ds.DeseqStats>`), among which:
#
# - ``alpha``: the p-value and adjusted p-value significance threshold (``0.05``
#   by default),
# - ``cooks_filter``: whether to filter p-values based on cooks outliers
#   (``True`` by default),
# - ``independent_filter``: whether to perform independent filtering to correct
#   p-value trends (``True`` by default).
#
# In the :ref:`section on multifactor analysis<multifactor_ref>`, we will also see how
# to use the ``contrast`` argument to specify according to which variable samples should
# be compared.

# %%
# .. _wald_ref:
#
# Wald test
# """"""""""
#
# PyDESeq2 computes p-values using Wald tests. This can be done using the
# :meth:`summary() <DeseqStats.summary>` method, which runs the whole statistical
# analysis, cooks filtering and multiple testing adjustement included.

stat_res.summary()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "stat_results.pkl"), "wb") as f:
        pkl.dump(stat_res, f)

# %%
# The results are then stored in the ``results_df`` attribute (``stat_res.results_df``).

# %%
# LFC shrinkage
# """""""""""""
#
# For visualization or post-processing purposes, it might be suitable to perform
# LFC shrinkage. This is implemented by the :meth:`lfc_shrink() <DeseqStats.lfc_shrink>`
# method.

stat_res.lfc_shrink(coeff="condition_B_vs_A")

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "shrunk_stat_results.pkl"), "wb") as f:
        pkl.dump(stat_res, f)

# %%
# .. note::
#   Running :meth:`lfc_shrink() <DeseqStats.lfc_shrink>` will overwrite a
#   :class:`DeseqStats`' log fold changes (and standard errors) with shrunk values.
#   This can be checked using the ``shrunk_LFCs`` flag.
#

print(stat_res.shrunk_LFCs)  # Will be True only if lfc_shrink() was run.

# %%
# .. _multifactor_ref:
#
# Multifactor analysis
# ---------------------
#
# .. currentmodule:: pydeseq2.dds
#
# So far, we have only used the ``condition`` column of ``clinical_df``, which divides
# samples between conditions ``A`` and ``B``. Yet, ``clinical_df`` contains second
# column, which separates samples according to ``group`` ``X`` and ``Y``.

print(clinical_df)

# %%
# The goal of multifactor analysis is to use *both* variables to fit LFCs.
#
# Read counts modeling
# ^^^^^^^^^^^^^^^^^^^^^
#
# To perform multifactor analysis with PyDESeq2, we start by inializing a
# :class:`DeseqDataSet` as previously, but we provide the list of variables we would like
# to use in the ``design_factors`` argument.
#

dds = DeseqDataSet(
    counts=counts_df,
    clinical=clinical_df,
    design_factors=["group", "condition"],
    refit_cooks=True,
    n_cpus=8,
)
# %%
# .. note::
#   By default, the last variable in the list (here, ``"condition"``) will be the one for
#   which LFCs and p-values will be displayed, but this may be changed later on when
#   performing the statistical analysis.
#
# As for the single-factor analysis, we fit dispersions and LFCs using the
# :meth:`deseq2() <DeseqDataSet.deseq2>` method.

dds.deseq2()

# %%
# Now, if we print log fold changes, we will have two columns in addition to the
# intercept: one corresponding to the ``group`` variable, and the other to ``condition``.

print(dds.varm["LFC"])

# %%
# .. currentmodule:: pydeseq2.ds
#
# Statistical analysis
# ^^^^^^^^^^^^^^^^^^^^
#
# P-values are computed as earlier from a :class:`DeseqStats` object with the
# :meth:`summary() <DeseqStats.summary>` method, with a new important argument:
# the ``contrast``.
# It is a list of three strings of the form
# ``["variable", "tested level", "reference level"]`` which determines which
# variable we want to compute LFCs and pvalues for.
# As an example, to compare the condition B to the condition A, we set
# ``contrast=["condition", "B", "A"]``.
#

stat_res_B_vs_A = DeseqStats(dds, contrast=["condition", "B", "A"], n_cpus=8)

# %%
# .. note::
#   If left blank, the variable of interest will be the last one provided in
#   the ``design_factors`` attribute of the corresponding
#   :class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>` object,
#   and the reference level will be picked alphabetically.
#   In any case, *both variables are still used*. This is due to the fact that ``dds``
#   was fit with both as design factors.
#
# Let us fit p-values:
#

stat_res_B_vs_A.summary()


# %%
# As we can see, although we are comparing the same cohorts (condition B vs A), the
# results differ from the :ref:`single-factor analysis <wald_ref>`. This is because the
# model uses information from both the ``condition`` and ``group`` variables.
#
# Let us now evaluate differential expression according to group Y vs X. To do so,
# we create a new :class:`DeseqStats` from the same
# :class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>`
# with ``contrast=["group", "Y", "X"]``, and run the analysis again.

stat_res_Y_vs_X = DeseqStats(dds, contrast=["group", "Y", "X"], n_cpus=8)
stat_res_Y_vs_X.summary()

# %%
# LFC shrinkage (multifactor)
# """""""""""""""""""""""""""
#
# In a multifactor setting, LFC shrinkage works as in the single-factor case, but will
# only shrink the LFCs of a :class:`DeseqStats` object based on its
# ``contrast`` argument.

stat_res_B_vs_A.lfc_shrink(coeff="condition_B_vs_A")
