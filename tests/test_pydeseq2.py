import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.preprocessing import deseq2_norm
from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform
from pydeseq2.utils import load_example_data

# Single-factor tests


@pytest.fixture
def counts_df():
    return load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )


@pytest.fixture
def metadata():
    return load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )


def test_size_factors_ratio(counts_df, metadata):
    """Test that the size_factors calcuation by ratio matches R."""

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors.csv"),
        index_col=0,
    )["x"].values

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds.fit_size_factors()

    np.testing.assert_almost_equal(
        dds.obsm["size_factors"].ravel(), r_size_factors.ravel()
    )


def test_size_factors_poscounts(counts_df, metadata):
    """Test that the size_factors calcuation by poscount matches R."""
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds.fit_size_factors("poscounts")

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors_poscount.csv"),
        index_col=0,
    )["sizeFactor"].values

    np.testing.assert_almost_equal(dds.obsm["size_factors"].ravel(), r_size_factors)


def test_size_factors_control_genes(counts_df, metadata):
    """Test that the size_factors calculation properly takes control_genes"""

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")

    dds.fit_size_factors(control_genes=["gene4"])

    np.testing.assert_almost_equal(
        dds.obsm["size_factors"].ravel(),
        counts_df["gene4"] / np.exp(np.log(counts_df["gene4"]).mean()),
    )

    dds.fit_size_factors(fit_type="poscounts", control_genes=[3])

    np.testing.assert_almost_equal(
        dds.obsm["size_factors"].ravel(),
        counts_df["gene4"] / np.exp(np.log(counts_df["gene4"]).mean()),
    )

    dds.fit_size_factors(fit_type="poscounts")

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        dds.obsm["size_factors"].ravel(),
        counts_df["gene4"] / np.exp(np.log(counts_df["gene4"]).mean()),
    )


def test_deseq_independent_filtering_parametric_fit(counts_df, metadata, tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        fit_type="parametric",
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_deseq_independent_filtering_mean_fit(counts_df, metadata, tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error, with a mean fit.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res_mean_curve.csv"),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        fit_type="mean",
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_deseq_without_independent_filtering_parametric_fit(
    counts_df, metadata, tol=0.02
):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error, with a parametric fit and no
    independent filtering.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(
            test_path, "data/single_factor/r_test_res_no_independent_filtering.csv"
        ),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        fit_type="parametric",
    )
    dds.deseq2()

    res = DeseqStats(dds, independent_filter=False)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


@pytest.mark.parametrize("alt_hypothesis", ["lessAbs", "greaterAbs", "less", "greater"])
def test_alt_hypothesis(alt_hypothesis, counts_df, metadata, tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error, with the alternative hypothesis test.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, f"data/single_factor/r_test_res_{alt_hypothesis}.csv"),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
    )
    dds.deseq2()

    res = DeseqStats(
        dds,
        lfc_null=-0.5 if alt_hypothesis == "less" else 0.5,
        alt_hypothesis=alt_hypothesis,
    )
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC and Wald statistics are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    if alt_hypothesis == "lessAbs":
        res_df.stat = res_df.stat.abs()
    assert (abs(r_res.stat - res_df.stat) / abs(r_res.stat)).max() < tol
    # Check for the same pvalue and padj where stat != 0
    assert (
        abs(r_res.pvalue[r_res.stat != 0] - res_df.pvalue[res_df.stat != 0])
        / r_res.pvalue[r_res.stat != 0]
    ).max() < tol


def test_deseq_no_refit_cooks(counts_df, metadata, tol=0.02):
    """Test that the outputs of the DESeq2 function *without cooks refit*
    match those of the original R package, up to a tolerance in relative error.
    Note: this is just to check that the workflow runs bug-free, as we expect no outliers
    in the synthetic dataset.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        refit_cooks=False,
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_lfc_shrinkage(counts_df, metadata, tol=0.02):
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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_dispersions.csv"),
        index_col=0,
    ).squeeze()

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition_B_vs_A")
    shrunk_res = res.results_df

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


def test_lfc_shrinkage_no_apeAdapt(counts_df, metadata, tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(
            test_path, "data/single_factor/r_test_lfc_shrink_no_apeAdapt_res.csv"
        ),
        index_col=0,
    )

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_dispersions.csv"),
        index_col=0,
    ).squeeze()

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition_B_vs_A", adapt=False)
    shrunk_res = res.results_df

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


def test_iterative_size_factors(counts_df, metadata, tol=0.02):
    """Test that the outputs of the iterative size factor method match those of the
    original R package (starting from the same inputs), up to a tolerance in relative
    error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_iterative_size_factors.csv"),
        index_col=0,
    ).squeeze()

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds._fit_iterate_size_factors()

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_size_factors - dds.obsm["size_factors"]) / abs(r_size_factors)
    ).max() < tol


# Multi-factor tests
@pytest.mark.parametrize("with_outliers", [True, False])
def test_multifactor_deseq(counts_df, metadata, with_outliers, tol=0.04):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    if with_outliers:
        r_res = pd.read_csv(
            os.path.join(test_path, "data/multi_factor/r_test_res_outliers.csv"),
            index_col=0,
        )
    else:
        r_res = pd.read_csv(
            os.path.join(test_path, "data/multi_factor/r_test_res.csv"),
            index_col=0,
        )

    if with_outliers:
        # Introduce outlier counts and a condition level with a single replicate
        counts_df.loc["sample1", "gene1"] = 2000
        counts_df.loc["sample11", "gene7"] = 1000
        metadata.loc["sample1", "condition"] = "C"

    dds = DeseqDataSet(
        counts=counts_df, metadata=metadata, design_factors=["group", "condition"]
    )
    dds.deseq2()

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_multifactor_lfc_shrinkage(counts_df, metadata, tol=0.02):
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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_dispersions.csv"), index_col=0
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df, metadata=metadata, design_factors=["group", "condition"]
    )
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition_B_vs_A")
    shrunk_res = res.results_df

    # Check that the same LFC found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


# Continuous tests
@pytest.mark.parametrize("with_outliers", [True, False])
def test_continuous_deseq(
    with_outliers,
    tol=0.04,
):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error, with a continuous factor.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    counts_df = pd.read_csv(
        os.path.join(test_path, "data/continuous/test_counts.csv"), index_col=0
    ).T

    metadata = pd.read_csv(
        os.path.join(test_path, "data/continuous/test_metadata.csv"), index_col=0
    )

    # Load R results
    if with_outliers:
        r_res = pd.read_csv(
            os.path.join(test_path, "data/continuous/r_test_res_outliers.csv"),
            index_col=0,
        )
    else:
        r_res = pd.read_csv(
            os.path.join(test_path, "data/continuous/r_test_res.csv"), index_col=0
        )

    if with_outliers:
        # Introduce outlier counts and a condition level with a single replicate
        counts_df.loc["sample1", "gene1"] = 2000
        counts_df.loc["sample11", "gene7"] = 1000
        metadata.loc["sample1", "condition"] = "C"

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors=["group", "condition", "measurement"],
        continuous_factors=["measurement"],
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_continuous_lfc_shrinkage(tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/continuous/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(test_path, "data/continuous/r_test_lfc_shrink_res.csv"),
        index_col=0,
    )

    counts_df = pd.read_csv(
        os.path.join(test_path, "data/continuous/test_counts.csv"), index_col=0
    ).T

    metadata = pd.read_csv(
        os.path.join(test_path, "data/continuous/test_metadata.csv"), index_col=0
    )

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/continuous/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/continuous/r_test_dispersions.csv"), index_col=0
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors=["group", "condition", "measurement"],
        continuous_factors=["measurement"],
    )

    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="measurement")
    shrunk_res = res.results_df

    # Check that the same LFC found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


@pytest.mark.parametrize("low_memory", [True, False])
def test_wide_deseq(
    low_memory,
    tol=0.02,
):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error, on a dataset with more genes than
    samples.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    counts_df = pd.read_csv(
        os.path.join(test_path, "data/wide/test_counts.csv"), index_col=0
    ).T

    metadata = pd.read_csv(
        os.path.join(test_path, "data/wide/test_metadata.csv"), index_col=0
    )

    # Load R results
    r_res = pd.read_csv(os.path.join(test_path, "data/wide/r_test_res.csv"), index_col=0)

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors=["group", "condition"],
        low_memory=low_memory,
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_contrast(counts_df, metadata):
    """
    Check that the contrasts ['condition', 'B', 'A'] and ['condition', 'A', 'B'] give
    coherent results (change of sign in LFCs and Wald stats, same (adjusted p-values).
    """

    dds = DeseqDataSet(
        counts=counts_df, metadata=metadata, design_factors=["group", "condition"]
    )
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


def test_anndata_init(counts_df, metadata, tol=0.02):
    """
    Test initializing dds with an AnnData object that already has filled in fields,
    including with the same names as those used by pydeseq2.
    """
    np.random.seed(42)

    # Make an anndata object
    adata = ad.AnnData(X=counts_df.astype(int), obs=metadata)

    # Put some dummy data in unused fields
    adata.obsm["dummy_metadata"] = np.random.choice(2, adata.n_obs)
    adata.varm["dummy_param"] = np.random.randn(adata.n_vars)

    # Put values in the dispersions field
    adata.varm["dispersions"] = np.random.randn(adata.n_vars) ** 2

    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    # Initialize DeseqDataSet from anndata and test results
    dds = DeseqDataSet(adata=adata, design_factors="condition")
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # Check results
    assert_res_almost_equal(res_df, r_res, tol)


def test_vst(counts_df, metadata, tol=0.02):
    """
    Test the output of VST compared with DESeq2.
    """
    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_vst = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_vst.csv"), index_col=0
    ).T

    r_vst_with_design = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_vst_with_design.csv"), index_col=0
    ).T

    # Test blind design
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors=["condition"])
    dds.vst(use_design=False)
    assert (np.abs(r_vst - dds.layers["vst_counts"]) / r_vst).max().max() < tol

    # Test full design
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors=["condition"])
    dds.vst(use_design=True)
    assert (
        np.abs(r_vst_with_design - dds.layers["vst_counts"]) / r_vst_with_design
    ).max().max() < tol


def test_mean_vst(counts_df, metadata, tol=0.02):
    """
    Test the output of VST with ``fitType="mean"`` compared with DESeq2.
    """
    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_vst = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_mean_vst.csv"), index_col=0
    ).T

    # Test blind design
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors=["condition"])
    dds.vst(use_design=False, fit_type="mean")
    assert (np.abs(r_vst - dds.layers["vst_counts"]) / r_vst).max().max() < tol


def test_ref_level(counts_df, metadata):
    """Test that DeseqDataSet columns are created according to the passed reference
    level, if any.
    """
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors=["group", "condition"],
        ref_level=["group", "Y"],
    )

    # Check that the column exists
    assert "group_X_vs_Y" in dds.obsm["design_matrix"].columns
    # Check that its content is correct
    assert (
        dds.obsm["design_matrix"]["group_X_vs_Y"]
        == metadata["group"].apply(
            lambda x: 1 if x == "X" else 0 if x == "Y" else np.nan
        )
    ).all()


def test_deseq2_norm(counts_df, metadata):
    """Test that deseq2_norm() called on a pandas dataframe outputs the same results as
    DeseqDataSet.fit_size_factors()
    """
    # Fit size factors from DeseqDataSet
    dds = DeseqDataSet(counts=counts_df, metadata=metadata)
    dds.fit_size_factors()
    s1 = dds.obsm["size_factors"]

    # Fit size factors from counts directly
    s2 = deseq2_norm(counts_df)[1]

    np.testing.assert_almost_equal(
        s1,
        s2,
        decimal=8,
    )


@pytest.fixture
def train_counts(counts_df):
    return counts_df[25:75]


@pytest.fixture
def test_counts(counts_df):
    return counts_df[0:25]


@pytest.fixture
def train_metadata(metadata):
    return metadata[25:75]


@pytest.fixture
def train_dds(train_counts, train_metadata):
    return DeseqDataSet(
        counts=train_counts, metadata=train_metadata, design_factors="condition"
    )


def test_deseq2_norm_fit(train_counts):
    # patient from 25 to 75
    logmeans, filtered_genes = deseq2_norm_fit(train_counts)

    # 10 genes
    assert logmeans.shape == (10,)
    assert filtered_genes.shape == (10,)


def test_deseq2_norm_transform(train_counts, test_counts):
    # First fit with some indices
    logmeans, filtered_genes = deseq2_norm_fit(train_counts)

    normed_counts, size_factors = deseq2_norm_transform(
        test_counts, logmeans, filtered_genes
    )
    assert isinstance(normed_counts, pd.DataFrame)
    # 25 samples, 10 genes
    assert normed_counts.shape == (25, 10)
    assert size_factors.shape == (25,)


def test_vst_fit(train_dds):
    # patient from 25 to 75
    train_dds.vst_fit()

    # the correct attributes are fit
    assert "vst_trend_coeffs" in train_dds.uns
    assert "normed_counts" in train_dds.layers
    assert "size_factors" in train_dds.obsm


def test_vst_transform(train_dds, test_counts):
    # First fit with some indices
    train_dds.vst_fit()

    result = train_dds.vst_transform(test_counts.to_numpy())
    assert isinstance(result, np.ndarray)
    # 25 samples, 10 genes
    assert result.shape == (25, 10)


@pytest.mark.parametrize(
    ("dea_fit_type", "vst_fit_type"),
    [
        ("mean", "parametric"),
        ("parametric", "mean"),
        ("parametric", "parametric"),
        ("mean", "mean"),
    ],
)
def test_vst_blind(train_counts, train_metadata, dea_fit_type, vst_fit_type):
    """Test vst with combinatory dea dea_fit_type and fit_type"""
    train_dds = DeseqDataSet(
        counts=train_counts,
        metadata=train_metadata,
        design_factors="condition",
        fit_type=dea_fit_type,
    )
    train_dds.deseq2()
    if dea_fit_type == "parametric":
        assert "trend_coeffs" in train_dds.uns
    else:
        assert "mean_disp" in train_dds.uns
    assert "normed_counts" in train_dds.layers
    assert "size_factors" in train_dds.obsm
    assert train_dds.fit_type == dea_fit_type

    train_dds.vst(use_design=False, fit_type=vst_fit_type)
    # Check that the dea fit type hasn't changed
    assert train_dds.fit_type == dea_fit_type


def test_vst_transform_no_fit(train_counts, train_metadata, test_counts):
    """Test vst_transform without calling vst_fit()"""
    train_dds = DeseqDataSet(
        counts=train_counts,
        metadata=train_metadata,
        design_factors="condition",
        fit_type="parametric",
    )
    with pytest.raises(RuntimeError):
        train_dds.vst_transform(test_counts.to_numpy())


def assert_res_almost_equal(py_res, r_res, tol=0.02):
    # check that the same p-values are NaN
    assert (py_res.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (py_res.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - py_res.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - py_res.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - py_res.padj) / r_res.padj).max() < tol
