"""
PyDESeq 2 pipeline
===========================

This notebook gives a minimalistic example of how to perform DEA using PyDESeq2.

It allows you to run the PyDESeq2 pipeline on synthetic data provided as part of this repository.
"""

import os
import pickle as pkl

from pydeseq2.DeseqDataSet import DeseqDataSet
from pydeseq2.DeseqStats import DeseqStats
from pydeseq2.utils import load_example_data

SAVE = False  # whether to save the outputs of this notebook

# %%
# Data loading
# ------------
#
# See the `datasets` readme for the required data organization.
#

DATASET = "synthetic"  # or 'TCGA-BRCA', 'TCGA-COAD', etc.

OUTPUT_PATH = f"../output_files/{DATASET}"
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

counts_df = load_example_data(
    modality="raw_counts",
    dataset=DATASET,
    debug=False,
)

clinical_df = load_example_data(
    modality="clinical",
    dataset=DATASET,
    debug=False,
)

print(counts_df)

# %%
# Remove samples for which `high_grade` is NaN.

if DATASET != "synthetic":
    samples_to_keep = ~clinical_df.high_grade.isna()
    samples_to_keep.sum()
    counts_df = counts_df.loc[samples_to_keep]
    clinical_df = clinical_df.loc[samples_to_keep]

# %%
# Filter out genes that have less than 10 counts in total

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
len(genes_to_keep)

counts_df = counts_df[genes_to_keep]

# %%
# 1 - Read counts modeling with the `DeseqDataSet` class
# ------------------------------------------------------
#
# We start by creating a `DeseqDataSet` object with the count and clinical data.
# Here, we use 8 threads, feel free to adapt this to your setup or to set
# to `None` to use all available CPUs.
#
# Another option of interest is the `refit_cooks` argument
# (set to `True` by default), which controls whether Cooks outlier
# should be refitted – this is advised, in general.
#
# .. note::
#  in the case of the provided synthetic data, there won't be any Cooks
#  outliers.
#
# Start by creating a DeseqDataSet
dds = DeseqDataSet(
    counts_df,
    clinical_df,
    design_factors="condition" if DATASET == "synthetic" else "high_grade",
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
# 2 - Statistical analysis with the `DeseqStats` class
# ----------------------------------------------------
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
# ^^^^^^^^^
#
# The `summary` function runs the statistical analysis (multiple testing
# adjustement included) and returns a summary DataFrame.

stat_res.summary()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "stat_results.pkl"), "wb") as f:
        pkl.dump(stat_res, f)

# %%
# LFC shrinkage
# ^^^^^^^^^^^^^
#
# For visualization or post-processing purposes, it might be suitable to perform
# LFC shrinkage. This is implemented by the `lfc_shrink` method.

stat_res.lfc_shrink()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "shrunk_stat_results.pkl"), "wb") as f:
        pkl.dump(stat_res, f)
