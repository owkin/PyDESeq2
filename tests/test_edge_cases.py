import numpy as np
import pandas as pd
import pytest

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data


def test_zero_genes():
    """Test that genes with only 0 counts are handled by DeseqDataSet et DeseqStats,
    and that NaNs are returned.
    """

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    n, m = counts_df.shape

    # Introduce a few genes with fully 0 counts
    np.random.seed(42)
    zero_genes = counts_df.columns[np.random.choice(m, size=m // 3, replace=False)]
    counts_df[zero_genes] = 0

    # Run analysis
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds.deseq2()

    # check that the corresponding parameters are NaN
    assert np.isnan(dds[:, zero_genes].varm["dispersions"]).all()
    assert np.isnan(dds[:, zero_genes].varm["LFC"]).all().all()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the corresponding stats are NaN
    assert (res_df.loc[zero_genes].baseMean == 0).all()
    assert res_df.loc[zero_genes].log2FoldChange.isna().all()
    assert res_df.loc[zero_genes].lfcSE.isna().all()
    assert res_df.loc[zero_genes].stat.isna().all()
    assert res_df.loc[zero_genes].pvalue.isna().all()
    assert res_df.loc[zero_genes].padj.isna().all()


# Tests on the count matrix
def test_nan_counts():
    """Test that a ValueError is thrown when the count matrix contains NaNs."""
    counts_df = pd.DataFrame(
        {"gene1": [0, np.nan], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")


def test_numeric_counts():
    """Test that a ValueError is thrown when the count matrix contains
    non-numeric values.
    """
    counts_df = pd.DataFrame(
        {"gene1": [0, "a"], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")


def test_integer_counts():
    """Test that a ValueError is thrown when the count matrix contains
    non-integer values."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1.5], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")


def test_non_negative_counts():
    """Test that a ValueError is thrown when the count matrix contains
    negative values."""
    counts_df = pd.DataFrame(
        {"gene1": [0, -1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")


# Tests on the metadata data (design factors)
def test_nan_factors():
    """Test that a ValueError is thrown when the design factor contains NaNs."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, np.NaN]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")


def test_one_factor():
    """Test that a ValueError is thrown when the design factor takes only one value."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 0]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")


def test_rank_deficient_design():
    """Test that a ValueError is thrown when the design matrix does not have full column
    rank."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame(
        {"condition": [0, 1], "batch": ["A", "B"]}, index=["sample1", "sample2"]
    )

    with pytest.warns(UserWarning):
        DeseqDataSet(
            counts=counts_df, metadata=metadata, design_factors=["condition", "batch"]
        )


def test_reference_level():
    """Test that a ValueError is thrown when the reference level is not one of the
    design factor values."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(KeyError):
        DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors="condition",
            ref_level="control",
        )


def test_lfc_shrinkage_coeff():
    """Test that a KeyError is thrown when attempting to shrink an unexisting LFC
    coefficient, but that setting coeff=None runs bug-free.
    """

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="condition")
    dds.deseq2()

    # Check that coeff=None gives the same results as the default
    # coefficient condition_B_vs_A
    res_1 = DeseqStats(dds)
    res_1.summary()
    res_1.lfc_shrink(coeff=None)
    shrunk_lfcs_1 = res_1.results_df["log2FoldChange"].copy()

    res_2 = DeseqStats(dds)
    res_2.summary()
    res_2.lfc_shrink(coeff="condition_B_vs_A")
    shrunk_lfcs_2 = res_2.results_df["log2FoldChange"].copy()

    assert (shrunk_lfcs_1 == shrunk_lfcs_2).all()

    with pytest.raises(KeyError):
        res_1.lfc_shrink(coeff="this_coeff_does_not_exist")


def test_indexes():
    """Test that a ValueError is thrown when the count matrix and the metadata data
    don't have the same index."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample01", "sample02"])

    with pytest.raises(ValueError):  # Should be raised by AnnData
        DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors="condition",
        )


def test_contrast():
    """Test that KeyErrors/ValueErrors are thrown when invalid contrasts are passed to a
    DeseqStats."""

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    # Run analysis
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        refit_cooks=False,
        design_factors=["condition", "group"],
    )
    dds.deseq2()

    # Too short
    with pytest.raises(ValueError):
        DeseqStats(dds, contrast=["condition", "B"])

    # Unexisting factor
    with pytest.raises(KeyError):
        DeseqStats(dds, contrast=["batch", "Y", "X"])

    # Unexisting factor reference level
    with pytest.raises(KeyError):
        DeseqStats(dds, contrast=["condition", "B", "C"])

    # Unexisting tested level
    with pytest.raises(KeyError):
        DeseqStats(dds, contrast=["condition", "C", "B"])


def test_cooks_not_refitted():
    """Test that an AttributeError is thrown when a `DeseqStats` object is initialized
    from a `DeseqDataSet` whose `refit_cooks` attribute is set to True, but whose
    Cooks outliers were not actually refitted."""

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    # Run analysis
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        refit_cooks=False,
        design_factors="condition",
    )
    dds.deseq2()

    # Set refit_cooks to True even though outliers were not refitted
    dds.refit_cooks = True

    with pytest.raises(AttributeError):
        stat_res = DeseqStats(dds)
        stat_res.summary()


def test_few_samples():
    """Test that PyDESeq2 runs bug-free on cohorts with less than 3 samples.
    (Check in particular that DeseqDataSet.calculate_cooks() runs).
    TODO: refine this test to check the correctness of the output?
    """

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    # Subsample two samples for each condition
    samples_to_keep = ["sample1", "sample2", "sample99", "sample100"]
    counts_df = counts_df.loc[samples_to_keep]
    metadata = metadata.loc[samples_to_keep]

    # Introduce an outlier
    counts_df.iloc[0, 0] = 1000

    # Run analysis. Should not throw degrees of freedom warning.
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        refit_cooks=True,
        design_factors="condition",
    )

    with pytest.warns(UserWarning):
        dds.deseq2()

    res = DeseqStats(dds)
    res.summary()

    # Check that no gene was refit, as there are not enough samples.
    assert dds.varm["replaced"].sum() == 0


def test_few_samples_and_outlier():
    """Test that PyDESeq2 runs bug-free with outliers and a cohort with less than 3
    samples.
    TODO: refine this test to check the correctness of the output?
    """

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    # Subsample two samples for each condition
    samples_to_keep = [
        "sample1",
        "sample2",
        "sample92",
        "sample93",
        "sample94",
        "sample95",
        "sample96",
        "sample97",
        "sample98",
        "sample99",
        "sample100",
    ]

    counts_df = counts_df.loc[samples_to_keep]
    metadata = metadata.loc[samples_to_keep]

    # Introduce outliers
    counts_df.iloc[0, 0] = 1000
    counts_df.iloc[-1, -1] = 1000

    # Run analysis. Should not throw an error.
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        refit_cooks=True,
        design_factors="condition",
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()


def test_zero_inflated():
    """
    Test the pydeseq2 throws a RuntimeWarning when there is at least one zero per gene.
    """

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    # Artificially zero-inflate the data
    # Each gene will have at least one sample with zero counts
    np.random.seed(42)
    idx = np.random.choice(len(counts_df), counts_df.shape[-1])
    counts_df.iloc[idx, :] = 0

    dds = DeseqDataSet(counts=counts_df, metadata=metadata)
    with pytest.warns(RuntimeWarning):
        dds.deseq2()


def test_plot_MA():
    """
    Test that a KeyError is thrown when attempting to run plot_MA without running the
    statistical analysis first.
    """

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )

    dds = DeseqDataSet(counts=counts_df, metadata=metadata)
    dds.deseq2()

    # Initialize a DeseqStats object without runnning the analysis
    res = DeseqStats(dds)

    with pytest.raises(AttributeError):
        res.plot_MA()

    # Run the analysis
    res.summary()
    # Now this shouldn't throw an error
    res.plot_MA()
