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
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.deseq2()

    # check that the corresponding parameters are NaN
    assert np.isnan(dds.var.loc[zero_genes, "dispersions"]).all()
    assert np.isnan(dds[:, zero_genes].varm["LFC"]).all().all()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()
    results_df = ds.results_df

    # check that the corresponding stats are NaN
    assert (results_df.loc[zero_genes].baseMean == 0).all()
    assert results_df.loc[zero_genes].log2FoldChange.isna().all()
    assert results_df.loc[zero_genes].lfcSE.isna().all()
    assert results_df.loc[zero_genes].stat.isna().all()
    assert results_df.loc[zero_genes].pvalue.isna().all()
    assert results_df.loc[zero_genes].padj.isna().all()


# Tests on the count matrix
def test_nan_counts():
    """Test that a ValueError is thrown when the count matrix contains NaNs."""
    counts_df = pd.DataFrame(
        {"gene1": [0, np.nan], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")


def test_numeric_counts():
    """Test that a ValueError is thrown when the count matrix contains
    non-numeric values.
    """
    counts_df = pd.DataFrame(
        {"gene1": [0, "a"], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")


def test_integer_counts():
    """Test that a ValueError is thrown when the count matrix contains
    non-integer values."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1.5], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")


def test_non_negative_counts():
    """Test that a ValueError is thrown when the count matrix contains
    negative values."""
    counts_df = pd.DataFrame(
        {"gene1": [0, -1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 1]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")


# Tests on the metadata data (design factors)
def test_nan_factors():
    """Test that a warning is thrown when the design factor contains NaNs."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, np.nan]}, index=["sample1", "sample2"])

    with pytest.raises(ValueError):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")


def test_one_factor():
    """Test that a ValueError is thrown when the design factor takes only one value."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame({"condition": [0, 0]}, index=["sample1", "sample2"])

    with pytest.warns(UserWarning):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")


def test_rank_deficient_design():
    """Test that a UserWarning is thrown when the design matrix does not have full
    column rank."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    metadata = pd.DataFrame(
        {"condition": [0, 1], "batch": ["A", "B"]}, index=["sample1", "sample2"]
    )

    with pytest.warns(UserWarning):
        DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition + batch")


def test_equal_num_vars_num_samples_design():
    """Test that a ValueError is thrown when fitting dispersions if the design matrix
    has equal numbers of rows and columns."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1, 55], "gene2": [4, 12, 60]},
        index=["sample1", "sample2", "sample3"],
    )
    metadata = pd.DataFrame(
        {"condition": [0, 1, 0], "batch": ["A", "B", "B"]},
        index=["sample1", "sample2", "sample3"],
    )

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition + batch")

    dds.fit_size_factors()

    with pytest.raises(ValueError):
        dds.fit_genewise_dispersions()


### Tests when a design matrix is passed directly


def test_matching_samples():
    """Test that a ValueError is thrown when the Index of the design matrix does not
    match obs."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1, 55], "gene2": [4, 12, 60]},
        index=["sample1", "sample2", "sample3"],
    )
    metadata = pd.DataFrame(
        {"condition": [0, 1, 0]},
        index=["sample1", "sample2", "sample3"],
    )

    with pytest.raises(ValueError):
        design_matrix = pd.DataFrame(
            {"intercept": [1.0, 1.0, 1.0], "condition": [0, 1, 0]},
            index=["sample1", "sample2", "sample5"],
        )
        DeseqDataSet(counts=counts_df, metadata=metadata, design=design_matrix)

    with pytest.raises(ValueError):
        design_matrix = pd.DataFrame(
            {"intercept": [1.0, 1.0], "condition": [0, 1]},
            index=["sample1", "sample2"],
        )
        DeseqDataSet(counts=counts_df, metadata=metadata, design=design_matrix)

    with pytest.raises(ValueError):
        design_matrix = pd.DataFrame(
            {"intercept": [1.0, 1.0, 1.0, 1.0], "condition": [0, 1, 0, 0]},
            index=["sample1", "sample2", "sample3", "sample4"],
        )
        DeseqDataSet(counts=counts_df, metadata=metadata, design=design_matrix)


def test_lfc_shrinkage_coeff():
    """Test that a KeyError is thrown when attempting to shrink an unexisting LFC
    coefficient.
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

    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition")
    dds.deseq2()

    # Check that coeff=None gives the same results as the default
    # coefficient condition_B_vs_A
    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    with pytest.raises(KeyError):
        ds.lfc_shrink(coeff="this_coeff_does_not_exist")


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
            design="~condition",
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
        design="~condition + group",
    )
    dds.deseq2()

    # Too short
    with pytest.raises(IndexError):
        DeseqStats(dds, contrast=["condition", "B"])

    # Unexisting factor
    with pytest.raises(ValueError):
        DeseqStats(dds, contrast=["batch", "Y", "X"])

    # Unexisting factor reference level
    with pytest.raises(ValueError):
        DeseqStats(dds, contrast=["condition", "B", "C"])

    # Unexisting tested level
    with pytest.raises(ValueError):
        DeseqStats(dds, contrast=["condition", "C", "B"])

    # Numerical contrast with the wrong number of elements
    with pytest.raises(ValueError):
        DeseqStats(dds, contrast=np.array([0, 0, 0, 1]))


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
        design="~condition",
    )
    dds.deseq2()

    # Set refit_cooks to True even though outliers were not refitted
    dds.refit_cooks = True

    with pytest.raises(AttributeError):
        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()


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
        design="~condition",
    )

    with pytest.warns(UserWarning):
        dds.deseq2()

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()

    # Check that no gene was refit, as there are not enough samples.
    assert dds.var["replaced"].sum() == 0


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
        design="~condition",
    )
    dds.deseq2()

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()


def test_new_all_zero_gene():
    """Test that a gene with only 0 counts after replacement is correctly handled."""
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

    # Subset samples for speed
    metadata = metadata.loc[[f"sample{i}" for i in [*range(1, 11), *range(91, 101)]]]
    counts_df = counts_df.loc[metadata.index]

    # Add a gene with a single outlier value and zeros elsewhere
    counts_df["geneX"] = 0
    counts_df.loc["sample100", "geneX"] = 100

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        refit_cooks=True,
    )

    with pytest.warns(UserWarning):
        # Will warn that parametric trend fit failed
        dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    # Check that the outlier gene leads to only zeros and that it is correctly handled
    # in DeseqStats
    assert dds.new_all_zeroes_genes.equals(pd.Index(["geneX"]))
    assert ds.results_df.loc["geneX", "baseMean"] == 0
    assert ds.results_df.loc["geneX", "log2FoldChange"] == 0
    assert ds.results_df.loc["geneX", "lfcSE"] == 0
    assert ds.results_df.loc["geneX", "stat"] == 0
    assert np.isnan(ds.results_df.loc["geneX", "pvalue"])
    assert np.isnan(ds.results_df.loc["geneX", "padj"])


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
    # Will warn that each gene contains a zero
    # and that the parametric curve is not a good fit
    with pytest.warns(UserWarning):
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
    ds = DeseqStats(dds, contrast=["condition", "B", "A"])

    with pytest.raises(AttributeError):
        ds.plot_MA()

    # Run the analysis
    ds.summary()
    # Now this shouldn't throw an error
    ds.plot_MA()
