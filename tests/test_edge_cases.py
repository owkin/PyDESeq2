import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tests
from pydeseq2.DeseqDataSet import DeseqDataSet
from pydeseq2.DeseqStats import DeseqStats


def test_zero_genes():
    """Test that genes with only 0 counts are handled by DeseqDataSet et DeseqStats,
    and that NaNs are returned.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    counts_df = pd.read_csv(
        os.path.join(test_path, "data/single_factor/test_counts.csv"), index_col=0
    ).T
    clinical_df = pd.read_csv(
        os.path.join(test_path, "data/single_factor/test_clinical.csv"), index_col=0
    )

    n, m = counts_df.shape

    # Introduce a few genes with fully 0 counts
    np.random.seed(42)
    zero_genes = counts_df.columns[np.random.choice(m, size=m // 3, replace=False)]
    counts_df[zero_genes] = 0

    # Run analysis
    dds = DeseqDataSet(counts_df, clinical_df, design_factors="condition")
    dds.deseq2()

    # check that the corresponding parameters are NaN
    assert dds.dispersions.loc[zero_genes].isna().all()
    assert dds.LFCs.loc[zero_genes].isna().all().all()

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
    counts_df = pd.DataFrame({"gene1": [0, np.nan], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, 1]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


def test_numeric_counts():
    """Test that a ValueError is thrown when the count matrix contains
    non-numeric values.
    """
    counts_df = pd.DataFrame({"gene1": [0, "a"], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, 1]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


def test_integer_counts():
    """Test that a ValueError is thrown when the count matrix contains
    non-integer values."""
    counts_df = pd.DataFrame({"gene1": [0, 1.5], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, 1]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


def test_non_negative_counts():
    """Test that a ValueError is thrown when the count matrix contains
    negative values."""
    counts_df = pd.DataFrame({"gene1": [0, -1], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, 1]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


# Tests on the clinical data (design factors)
def test_nan_factors():
    """Test that a ValueError is thrown when the design factor contains NaNs."""
    counts_df = pd.DataFrame({"gene1": [0, 1], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, np.NaN]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


def test_one_factors():
    """Test that a ValueError is thrown when the design factor takes only one value ."""
    counts_df = pd.DataFrame({"gene1": [0, 1], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, 0]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


def test_too_many_factors():
    """Test that a ValueError is thrown when the design factor takes
    more than two values."""
    counts_df = pd.DataFrame({"gene1": [0, 1, 5], "gene2": [4, 12, 8]})
    clinical_df = pd.DataFrame({"condition": [0, 1, 2]})

    with pytest.raises(ValueError):
        DeseqDataSet(counts_df, clinical_df, design_factors="condition")


def test_reference_level():
    """Test that a ValueError is thrown when the reference level is not one of the
    design factor values."""
    counts_df = pd.DataFrame({"gene1": [0, 1], "gene2": [4, 12]})
    clinical_df = pd.DataFrame({"condition": [0, 1]})

    with pytest.raises(KeyError):
        DeseqDataSet(
            counts_df, clinical_df, design_factors="condition", reference_level="control"
        )


def test_indexes():
    """Test that a ValueError is thrown when the count matrix and the clinical data
    don't have the same index."""
    counts_df = pd.DataFrame(
        {"gene1": [0, 1], "gene2": [4, 12]}, index=["sample1", "sample2"]
    )
    clinical_df = pd.DataFrame({"condition": [0, 1]}, index=["sample01", "sample02"])

    with pytest.raises(ValueError):
        DeseqDataSet(
            counts_df,
            clinical_df,
            design_factors="condition",
        )
