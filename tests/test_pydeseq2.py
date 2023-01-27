import os
from pathlib import Path

import numpy as np
import pandas as pd

import tests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

# Single-factor tests


def test_deseq(tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

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

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(counts_df, clinical_df, design_factors="condition")
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_deseq_no_refit_cooks(tol=0.02):
    """Test that the outputs of the DESeq2 function *without cooks refit*
    match those of the original R package, up to a tolerance in relative error.
    Note: this is just to check that the workflow runs bug-free, as we expect no outliers
    in the synthetic dataset.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

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

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(
        counts_df, clinical_df, design_factors="condition", refit_cooks=False
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_lfc_shrinkage(tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_lfc_shrink_res.csv"),
        index_col=0,
    )

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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_dispersions.csv"), index_col=0
    ).squeeze()

    dds = DeseqDataSet(counts_df, clinical_df, design_factors="condition")
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink()
    shrunk_res = res.results_df

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


# Multi-factor tests


def test_multifactor_deseq(tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

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

    r_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(counts_df, clinical_df, design_factors=["group", "condition"])
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_multifactor_lfc_shrinkage(tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_lfc_shrink_res.csv"),
        index_col=0,
    )

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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_dispersions.csv"), index_col=0
    ).squeeze()

    dds = DeseqDataSet(counts_df, clinical_df, design_factors=["group", "condition"])
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink()
    shrunk_res = res.results_df

    # Check that the same LFC found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


def test_contrast():
    """
    Check that the contrasts ['condition', 'B', 'A'] and ['condition', 'A', 'B'] give
    coherent results (change of sign in LFCs and Wald stats, same (adjusted p-values).
    """

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

    dds = DeseqDataSet(counts_df, clinical_df, design_factors=["group", "condition"])
    dds.deseq2()

    res_B_vs_A = DeseqStats(dds, contrast=["condition", "B", "A"])
    res_A_vs_B = DeseqStats(dds, contrast=["condition", "A", "B"])
    res_B_vs_A.summary()
    res_A_vs_B.summary()

    # Check that all values correspond, up to signs
    for col in res_B_vs_A.results_df.columns:
        np.testing.assert_almost_equal(
            res_B_vs_A.results_df[col].abs().values,
            res_A_vs_B.results_df[col].abs().values,
            decimal=8,
        )

    # Check that the sign of LFCs and stats are inverted
    np.testing.assert_almost_equal(
        res_B_vs_A.results_df.log2FoldChange.values,
        -res_A_vs_B.results_df.log2FoldChange.values,
        decimal=8,
    )
    np.testing.assert_almost_equal(
        res_B_vs_A.results_df.stat.values, -res_A_vs_B.results_df.stat.values, decimal=8
    )
