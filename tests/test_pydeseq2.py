import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix

import tests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.preprocessing import deseq2_norm
from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform
from pydeseq2.utils import load_example_data


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


# Single-factor tests


def test_size_factors_ratio(counts_df, metadata):
    """Test that the size_factors calcuation by ratio matches R."""

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors.csv"),
        index_col=0,
    )["x"].values

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.fit_size_factors()

    np.testing.assert_array_almost_equal(dds.obs["size_factors"], r_size_factors)


def test_size_factors_poscounts(counts_df, metadata):
    """Test that the size_factors calcuation by poscount matches R."""
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.fit_size_factors("poscounts")

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors_poscount.csv"),
        index_col=0,
    )["sizeFactor"].values

    np.testing.assert_array_almost_equal(dds.obs["size_factors"], r_size_factors)


def test_size_factors_control_genes(counts_df, metadata):
    """Test that the size_factors calculation properly takes control_genes"""

    dds = DeseqDataSet(
        counts=counts_df, metadata=metadata, design="~condition", control_genes=["gene4"]
    )

    dds.fit_size_factors()

    np.testing.assert_array_almost_equal(
        dds.obs["size_factors"],
        counts_df["gene4"] / np.exp(np.log(counts_df["gene4"]).mean()),
    )
    # We should have gene4 as control gene again.
    dds.fit_size_factors(fit_type="poscounts")

    # Gene 4 has no zero counts, so we should get the same as before
    np.testing.assert_array_almost_equal(
        dds.obs["size_factors"],
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
        design="~condition",
        fit_type="parametric",
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


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
        design="~condition",
        fit_type="mean",
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


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
        design="~condition",
        fit_type="parametric",
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"], independent_filter=False)
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


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
        counts=counts_df, metadata=metadata, design="~condition", n_cpus=2
    )
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        lfc_null=-0.5 if alt_hypothesis == "less" else 0.5,
        alt_hypothesis=alt_hypothesis,
        n_cpus=2,
    )
    ds.summary()

    # check that the same p-values are NaN
    assert (ds.results_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (ds.results_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC and Wald statistics are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - ds.results_df.log2FoldChange)
        / abs(r_res.log2FoldChange)
    ).max() < tol
    if alt_hypothesis == "lessAbs":
        ds.results_df.stat = ds.results_df.stat.abs()
    assert (abs(r_res.stat - ds.results_df.stat) / abs(r_res.stat)).max() < tol
    # Check for the same pvalue and padj where stat != 0
    assert (
        abs(
            r_res.pvalue[r_res.stat != 0] - ds.results_df.pvalue[ds.results_df.stat != 0]
        )
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
        design="~condition",
        refit_cooks=False,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


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

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.deseq2()
    dds.obs["size_factors"] = r_size_factors.values
    dds.var["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition[T.B]")
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

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.deseq2()
    dds.obs["size_factors"] = r_size_factors.values
    dds.var["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition[T.B]", adapt=False)
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

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds._fit_iterate_size_factors()

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_size_factors.values - dds.obs["size_factors"].values)
        / abs(r_size_factors.values)
    ).max() < tol


def test_lfc_shrinkage_large_counts(counts_df, metadata, tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error in
    the presence of a gene with very large counts.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/large_counts/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(test_path, "data/large_counts/r_test_lfc_shrink_res.csv"),
        index_col=0,
    )

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/large_counts/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/large_counts/r_test_dispersions.csv"),
        index_col=0,
    ).squeeze()

    counts_df = pd.DataFrame(
        data=[
            [25, 405, 1355, 12558, 489843],
            [28, 480, 2144, 13844, 514571],
            [12, 690, 1919, 15632, 564106],
            [31, 420, 1684, 11513, 556380],
            [34, 278, 3849, 11577, 412551],
            [19, 249, 3086, 7296, 295565],
            [17, 491, 4089, 13805, 280945],
            [15, 251, 2785, 10492, 214062],
        ],
        index=["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
        columns=["g1", "g2", "g3", "g4", "g5"],
    )

    metadata_df = pd.DataFrame(
        data=["A", "A", "A", "A", "B", "B", "B", "B"],
        index=["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
        columns=["condition"],
    )

    dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design="~condition")
    dds.deseq2()

    dds.obs["size_factors"] = r_size_factors.values
    dds.var["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition[T.B]")
    shrunk_res = res.results_df

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
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

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
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

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()
    dds.obs["size_factors"] = r_size_factors.values
    dds.var["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition[T.B]")
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
        design="~group + condition + measurement",
    )
    dds.deseq2()

    contrast_vector = np.zeros(dds.obsm["design_matrix"].shape[1])
    contrast_vector[-1] = 1

    ds = DeseqStats(dds, contrast=contrast_vector)
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


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
        design="~group + condition + measurement",
    )
    dds.deseq2()

    contrast_vector = np.zeros(dds.obsm["design_matrix"].shape[1])
    contrast_vector[-1] = 1

    dds.obs["size_factors"] = r_size_factors.values
    dds.var["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds, contrast=contrast_vector)
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
        design="~group + condition",
        low_memory=low_memory,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


def test_contrast(counts_df, metadata):
    """
    Check that the contrasts ['condition', 'B', 'A'] and ['condition', 'A', 'B'] give
    coherent results (change of sign in LFCs and Wald stats, same (adjusted p-values).
    """

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()

    res_B_vs_A = DeseqStats(dds, contrast=["condition", "B", "A"])
    res_A_vs_B = DeseqStats(dds, contrast=["condition", "A", "B"])
    res_B_vs_A.summary()
    res_A_vs_B.summary()

    # Check that all values correspond, up to signs
    for col in res_B_vs_A.results_df.columns:
        np.testing.assert_array_almost_equal(
            res_B_vs_A.results_df[col].abs().values,
            res_A_vs_B.results_df[col].abs().values,
            decimal=8,
        )

    # Check that the sign of LFCs and stats are inverted
    np.testing.assert_array_almost_equal(
        res_B_vs_A.results_df.log2FoldChange.values,
        -res_A_vs_B.results_df.log2FoldChange.values,
        decimal=8,
    )
    np.testing.assert_array_almost_equal(
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
    adata.var["dummy_param"] = np.random.randn(adata.n_vars)

    # Put values in the dispersions field
    adata.var["dispersions"] = np.random.randn(adata.n_vars) ** 2

    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    # Initialize DeseqDataSet from anndata and test results
    dds = DeseqDataSet(adata=adata, design="~condition")
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


def test_design_matrix_init(counts_df, metadata, tol=0.02):
    """
    Test initializing dds with design matrix.
    """
    np.random.seed(42)

    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    # Create a design matrix
    design_matrix = pd.DataFrame(model_matrix("~condition", metadata))

    # Simulate the user picking a different column name
    design_matrix.rename(columns={"condition[T.B]": "condition_B"}, inplace=True)

    # Initialize DeseqDataSet from design matrix and test results
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design=design_matrix)
    dds.deseq2()

    ds = DeseqStats(dds, contrast=np.array([0, 1]))
    ds.summary()

    # Check results
    assert_res_almost_equal(ds.results_df, r_res, tol)


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
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.vst(use_design=False)
    assert (np.abs(r_vst - dds.layers["vst_counts"]) / r_vst).max().max() < tol

    # Test full design
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
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
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.vst(use_design=False, fit_type="mean")
    assert (np.abs(r_vst - dds.layers["vst_counts"]) / r_vst).max().max() < tol


def test_deseq2_norm(counts_df, metadata):
    """Test that deseq2_norm() called on a pandas dataframe outputs the same results as
    DeseqDataSet.fit_size_factors()
    """
    # Fit size factors from DeseqDataSet
    dds = DeseqDataSet(counts=counts_df, metadata=metadata)
    dds.fit_size_factors()
    s1 = dds.obs["size_factors"]

    # Fit size factors from counts directly
    s2 = deseq2_norm(counts_df)[1]

    np.testing.assert_array_almost_equal(
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
        counts=train_counts, metadata=train_metadata, design="~condition"
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
    assert "size_factors" in train_dds.obs


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
        design="~condition",
        fit_type=dea_fit_type,
    )
    train_dds.deseq2()
    if dea_fit_type == "parametric":
        assert "trend_coeffs" in train_dds.uns
    else:
        assert "mean_disp" in train_dds.uns
    assert "normed_counts" in train_dds.layers
    assert "size_factors" in train_dds.obs
    assert train_dds.fit_type == dea_fit_type

    train_dds.vst(use_design=False, fit_type=vst_fit_type)
    # Check that the dea fit type hasn't changed
    assert train_dds.fit_type == dea_fit_type


def test_vst_transform_no_fit(train_counts, train_metadata, test_counts):
    """Test vst_transform without calling vst_fit()"""
    train_dds = DeseqDataSet(
        counts=train_counts,
        metadata=train_metadata,
        design="~condition",
        fit_type="parametric",
    )
    with pytest.raises(RuntimeError):
        train_dds.vst_transform(test_counts.to_numpy())


# Likelihood ratio test (LRT)


@pytest.mark.parametrize(
    "design, reduced, ref_file",
    [
        ("~condition", "~1", "data/single_factor/r_test_res_lrt.csv"),
        ("~group + condition", "~group", "data/multi_factor/r_test_res_lrt.csv"),
        (
            "~group + condition",
            "~1",
            "data/multi_factor/r_test_res_lrt_reduced_intercept.csv",
        ),
    ],
)
def test_lrt_matches_r(counts_df, metadata, design, reduced, ref_file):
    """The likelihood ratio test matches R DESeq2's ``DESeq(dds, test="LRT")``.

    Covers a single-factor model (``df=1``), a multi-factor model testing one
    factor adjusting for another (``df=1``), and a joint test of both factors
    (``df=2``).
    """
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(os.path.join(test_path, ref_file), index_col=0)

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design=design)
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"], test="LRT", reduced=reduced)
    ds.summary()

    assert_lrt_res_almost_equal(ds.results_df, r_res)


def test_lrt_no_independent_filtering(counts_df, metadata):
    """The raw LRT statistic and p-value match R with independent filtering off."""
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(
            test_path, "data/multi_factor/r_test_res_lrt_no_independent_filtering.csv"
        ),
        index_col=0,
    )

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~group",
        independent_filter=False,
    )
    ds.summary()

    assert_lrt_res_almost_equal(ds.results_df, r_res)


def test_lrt_with_outliers(counts_df, metadata, tol=0.04):
    """The LRT matches R when Cooks outliers are replaced and models refitted.

    Uses the same mild outlier injection as the Wald outlier test. The outlier
    counts are replaced and both the full and reduced models are refitted on the
    imputed counts, matching R's ``counts(dds, replaced=TRUE)`` behaviour.
    """
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_res_lrt_outliers.csv"),
        index_col=0,
    )

    counts_df.loc["sample1", "gene1"] = 2000
    counts_df.loc["sample11", "gene7"] = 1000

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()

    # At least one gene should have been flagged as an outlier and refitted.
    assert dds.var["replaced"].sum() > 0

    counts_before = dds.X.copy()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"], test="LRT", reduced="~group")
    ds.summary()

    # The LRT must not mutate the raw counts when reconstructing replaced counts.
    np.testing.assert_array_equal(dds.X, counts_before)

    assert_lrt_res_almost_equal(ds.results_df, r_res, tol=tol)


def test_lrt_reduced_as_design_matrix(counts_df, metadata):
    """A reduced model given as a design matrix matches the equivalent formula."""
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()

    ds_formula = DeseqStats(
        dds, contrast=["condition", "B", "A"], test="LRT", reduced="~group"
    )
    ds_formula.summary()

    # Build the same reduced design as an explicit matrix.
    reduced_matrix = model_matrix("~group", metadata).to_numpy()
    ds_matrix = DeseqStats(
        dds, contrast=["condition", "B", "A"], test="LRT", reduced=reduced_matrix
    )
    ds_matrix.summary()

    pd.testing.assert_series_equal(
        ds_formula.statistics, ds_matrix.statistics, check_names=False
    )
    pd.testing.assert_series_equal(
        ds_formula.p_values, ds_matrix.p_values, check_names=False
    )


def test_lrt_statistic_definition(counts_df, metadata):
    """The LRT is internally consistent, independent of the R reference.

    Checks that the statistic is non-negative, that the p-value is exactly the
    chi-squared survival function of the statistic, and that for a single removed
    coefficient (``df=1``) the LRT statistic is close to the squared Wald
    statistic (their asymptotic relationship) for genes with signal.
    """
    from scipy.stats import chi2

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.deseq2()

    ds_lrt = DeseqStats(dds, contrast=["condition", "B", "A"], test="LRT", reduced="~1")
    ds_lrt.summary()

    stat = ds_lrt.statistics.dropna()
    # Statistic is non-negative.
    assert (stat >= 0).all()
    # p-value is the chi2 survival function of the statistic (df = 1 here).
    df = ds_lrt.design_matrix.shape[1] - ds_lrt.reduced_design_matrix.shape[1]
    assert df == 1
    np.testing.assert_allclose(
        ds_lrt.p_values.dropna().values, chi2.sf(stat.values, df), rtol=1e-10
    )

    # LRT statistic ~ (Wald statistic)^2 for genes with signal (df = 1).
    ds_wald = DeseqStats(dds, contrast=["condition", "B", "A"], test="Wald")
    ds_wald.summary()
    signal = ds_wald.p_values < 0.1
    ratio = (stat[signal] / ds_wald.statistics[signal] ** 2).dropna()
    assert ((ratio > 0.9) & (ratio < 1.1)).all()


def test_lrt_lfc_matches_wald(counts_df, metadata):
    """The LRT reports the same LFC as the Wald test (both use the full model)."""
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()

    ds_wald = DeseqStats(dds, contrast=["condition", "B", "A"], test="Wald")
    ds_wald.summary()

    ds_lrt = DeseqStats(
        dds, contrast=["condition", "B", "A"], test="LRT", reduced="~group"
    )
    ds_lrt.summary()

    pd.testing.assert_series_equal(
        ds_wald.results_df["log2FoldChange"],
        ds_lrt.results_df["log2FoldChange"],
        check_names=False,
    )


def test_lrt_input_validation(counts_df, metadata):
    """Invalid ``test``/``reduced`` combinations raise informative errors."""
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~group + condition")
    dds.deseq2()

    # LRT requires a reduced model.
    with pytest.raises(ValueError, match="reduced"):
        DeseqStats(dds, contrast=["condition", "B", "A"], test="LRT")

    # reduced is not allowed for the Wald test.
    with pytest.raises(ValueError, match="reduced"):
        DeseqStats(dds, contrast=["condition", "B", "A"], test="Wald", reduced="~group")

    # Unknown test.
    with pytest.raises(ValueError, match="test"):
        DeseqStats(dds, contrast=["condition", "B", "A"], test="chi2")

    # Reduced model not smaller than the full model.
    with pytest.raises(ValueError, match="fewer coefficients"):
        DeseqStats(
            dds,
            contrast=["condition", "B", "A"],
            test="LRT",
            reduced="~group + condition",
        )

    # Reduced model not nested in the full model (a factor absent from the design).
    dds_single = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds_single.deseq2()
    with pytest.raises(ValueError, match="nested"):
        DeseqStats(
            dds_single, contrast=["condition", "B", "A"], test="LRT", reduced="~group"
        )


def assert_lrt_res_almost_equal(py_res, r_res, tol=0.02, stat_atol=0.1, pval_tol=0.03):
    """Compare PyDESeq2 LRT results to an R DESeq2 reference.

    The full-model log-fold changes match R tightly. The LRT statistic matches
    with a relative tolerance for genes with signal and an absolute tolerance for
    near-null genes, whose statistic is ~0 and therefore dominated by IRLS
    convergence noise (in R as well as here). For genes R deems differentially
    expressed, p-values and adjusted p-values match R; very small p-values may
    differ by a few percent because the chi-squared tail amplifies tiny
    differences in the statistic.
    """
    # Same genes flagged NaN (Cooks / all-zero filtering).
    assert (py_res.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (py_res.padj.isna() == r_res.padj.isna()).all()

    # Full-model LFCs.
    assert (
        abs(r_res.log2FoldChange - py_res.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol

    # LRT statistics.
    common = py_res.stat.dropna().index
    np.testing.assert_allclose(
        py_res.stat.loc[common].values,
        r_res.stat.loc[common].values,
        rtol=tol,
        atol=stat_atol,
    )

    # p-values / adjusted p-values for genes R deems significant.
    signal = r_res.pvalue < 0.1
    assert (abs(r_res.pvalue - py_res.pvalue) / r_res.pvalue)[signal].max() < pval_tol
    assert (abs(r_res.padj - py_res.padj) / r_res.padj)[signal].max() < pval_tol


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
