"""
Step-by-step PyDESeq2 workflow
===============================

This notebook details all the steps of the PyDESeq2 pipeline.

It allows you to run the PyDESeq2 pipeline on the synthetic data provided
in this repository.

If this is your first contact with PyDESeq2, we recommend you first have a look at the
:doc:`standard workflow example <plot_minimal_pydeseq2_pipeline>`.

.. contents:: Contents
    :local:
    :depth: 3

We start by importing required packages and setting up an optional path to save
results.
"""

import os
import pickle as pkl

import anndata as ad

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

SAVE = False  # whether to save the outputs of this notebook

if SAVE:
    # Replace this with the path to directory where you would like results to be
    # saved
    OUTPUT_PATH = "../output_files/synthetic_example"
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# %%
# Data loading
# ------------
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
# To illustrate the required data format, we load a synthetic example dataset
# that may be
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
# As in the :doc:`standard workflow example <plot_minimal_pydeseq2_pipeline>`,
# the clinical data contains two columns, ``condition`` and ``group``,
# representing two types of bi-level annotations. In this example, we will only use the
# ``condition`` factor.
# To see how to use both the ``condition`` and the ``group`` factors in the analysis,
# see the :ref:`section on multifactor analysis<multifactor_ref>` from the first example.

# %%
# Data filtering
# ^^^^^^^^^^^^^^
#
# As in the :doc:`standard workflow example <plot_minimal_pydeseq2_pipeline>`,
# we filter out genes with low total read counts and samples with missing
# information before proceeding with DEA.

samples_to_keep = ~clinical_df.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
clinical_df = clinical_df.loc[samples_to_keep]
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# %%
# Now that we have loaded and filtered our data, we may proceed with the differential
# analysis.

# %%
# .. currentmodule:: pydeseq2.dds
#
# 1. Read counts modeling with the :class:`DeseqDataSet` class
# -------------------------------------------------------------
# We start by creating a :class:`DeseqDataSet`
# object from the count and clinical data.
# A :class:`DeseqDataSet` fits dispersion and
# log-fold change (LFC) parameters from the data, and stores them.
#
# A :class:`DeseqDataSet` may be either be initialized from a ``counts`` and
# a ``clinical`` dataframe (see :ref:`the first example <standard_ex_dds_ref>`) or,
# as illustrated here, an  :class:`AnnData <anndata.AnnData>` object, which may contain
# information from a prior analysis.
#
adata = ad.AnnData(X=counts_df, obs=clinical_df)

print(adata)

# %%
dds = DeseqDataSet(
    adata=adata,
    design_factors="condition",  # compare samples based on the "condition"
    # column ("B" vs "A")
    refit_cooks=True,
    n_cpus=8,
)

# %%
# The :class:`DeseqDataSet` class has two mandatory
# arguments, `counts_df` and
# `clinical_df`, as well as a set of optional keyword arguments, among which:
#
# - `design_factor`: the name of the column of clinical to be used as a design
#   variable
# - `refit_cooks`: whether to refit cooks outliers – this is advised, in general.
#
# .. note::
#   in the case of the provided synthetic data, there won't be any Cooks
#   outliers.

# %%
# Compute normalization factors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dds.fit_size_factors()

dds.obsm["size_factors"]

# %%
# Fit genewise dispersions
# ^^^^^^^^^^^^^^^^^^^^^^^^

dds.fit_genewise_dispersions()

dds.varm["genewise_dispersions"]

# %%
# Fit dispersion trend coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dds.fit_dispersion_trend()
dds.uns["trend_coeffs"]
dds.varm["fitted_dispersions"]

# %%
# Dispersion priors
# ^^^^^^^^^^^^^^^^^

dds.fit_dispersion_prior()
print(
    f"logres_prior={dds.uns['_squared_logres']}, sigma_prior={dds.uns['prior_disp_var']}"
)

# %%
# MAP Dispersions
# ^^^^^^^^^^^^^^^
# The `fit_MAP_dispersions` method filters the genes for which dispersion
# shrinkage is applied.
# Indeed, for genes whose MLE dispersions are too high above the trend curve,
# the original MLE value is kept.
# The final values of the dispersions that are used for downstream analysis is
# stored in `dds.dispersions`.

dds.fit_MAP_dispersions()
dds.varm["MAP_dispersions"]
dds.varm["dispersions"]

# %%
# Fit log fold changes
# ^^^^^^^^^^^^^^^^^^^^
# Note that in the `DeseqDataSet` object, the log-fold changes are stored in
# natural
# log scale, but that the results dataframe output by the `summary` method of
# `DeseqStats` displays LFCs in log2 scale (see later on).

dds.fit_LFC()
dds.varm["LFC"]

# %%
# Calculate Cooks distances and refit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that this step is optional.

dds.calculate_cooks()
if dds.refit_cooks:
    # Replace outlier counts
    dds.refit()

# %%
# Save everything

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "dds_detailed_pipe.pkl"), "wb") as f:
        pkl.dump(dds, f)

# %%
# .. currentmodule:: pydeseq2.ds
#
# 2. Statistical analysis
# -----------------------
# Statistical analysis with the :class:`DeseqStats` class.
# The `DeseqDataSet` class has a unique mandatory arguments, `dds`, which should
# be a *fitted* `DeseqDataSet` object, as well as a set of optional keyword
# arguments, among which:
#
# - `alpha`: the p-value and adjusted p-value significance threshold
# - `cooks_filter`: whether to filter p-values based on cooks outliers
# - `independent_filter`: whether to perform independent filtering to correct
#   p-value trends.

stat_res = DeseqStats(dds, alpha=0.05, cooks_filter=True, independent_filter=True)

# %%
# Wald tests
# ^^^^^^^^^^

stat_res.run_wald_test()
stat_res.p_values

# %%
# Cooks filtering
# ^^^^^^^^^^^^^^^
# This is optional
#
# .. note::
#   Note: in the case of the provided synthetic data, there won't be any
#   outliers.

if stat_res.cooks_filter:
    stat_res._cooks_filtering()
stat_res.p_values

# %%
# P-value adjustment
# ^^^^^^^^^^^^^^^^^^

if stat_res.independent_filter:
    stat_res._independent_filtering()
else:
    stat_res._p_value_adjustment()

stat_res.padj

# %%
# Building a results dataframe
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This dataframe is stored in the `results_df` attribute of the `DeseqStats`
# class.

stat_res.summary()

# %%
# Save everything if SAVE is set to True

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "stat_results_detailed_pipe.pkl"), "wb") as f:
        pkl.dump(stat_res, f)

# %%
# LFC Shrinkage
# ^^^^^^^^^^^^^
# For visualization or post-processing purposes, it might be suitable to perform
# LFC shrinkage. This is implemented by the `lfc_shrink` method.

stat_res.lfc_shrink(coeff="condition_B_vs_A")


# %%
# Save everything

if SAVE:
    with open(
        os.path.join(OUTPUT_PATH, "shrunk_stat_results_detailed_pipe.pkl"), "wb"
    ) as f:
        pkl.dump(stat_res, f)
