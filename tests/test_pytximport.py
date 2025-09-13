import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


@pytest.fixture
def pytximport_adata():
    """Load test pytximport AnnData object."""
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    adata_path = os.path.join(test_path, "data/pytximport/test_pytximport.h5ad")
    return ad.read_h5ad(adata_path)


@pytest.fixture
def standard_adata():
    """Create standard AnnData object without pytximport fields."""
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    counts_path = os.path.join(test_path, "data/pytximport/test_counts.csv")
    metadata_path = os.path.join(test_path, "data/pytximport/test_metadata.csv")

    counts_df = pd.read_csv(counts_path, index_col=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    adata = ad.AnnData(X=counts_df.T.values.round(), obs=metadata_df)
    adata.var_names = counts_df.index
    adata.obs_names = counts_df.columns

    return adata


class TestPytximportDetection:
    """Test pytximport data detection and validation."""

    def test_explicit_flag(self, pytximport_adata):
        """Test explicit from_pytximport=True flag."""
        dds = DeseqDataSet(
            adata=pytximport_adata, design="~condition", from_pytximport=True
        )

        # Should have detected pytximport data during initialization
        assert dds.from_pytximport is True

        dds.fit_size_factors()
        assert "normalization_factors" in dds.obsm
        assert dds.obsm["normalization_factors"].shape == (4, 41127)

    def test_no_pytximport_data(self, standard_adata):
        """Test standard DESeq2 without pytximport data."""
        dds = DeseqDataSet(adata=standard_adata, design="~condition")

        dds.fit_size_factors()
        assert "normalization_factors" not in dds.obsm
        assert "size_factors" in dds.obs


class TestPytximportValidation:
    """Test pytximport data validation."""

    def test_missing_length_matrix(self, standard_adata):
        """Test validation when length matrix is missing."""
        with pytest.raises(ValueError, match="'length' not found in obsm"):
            DeseqDataSet(adata=standard_adata, design="~condition", from_pytximport=True)

    def test_missing_counts_from_abundance(self, pytximport_adata):
        """Test validation when counts_from_abundance is missing."""
        # Remove the counts_from_abundance key
        del pytximport_adata.uns["counts_from_abundance"]

        with pytest.raises(ValueError, match="'counts_from_abundance' not found in uns"):
            DeseqDataSet(
                adata=pytximport_adata, design="~condition", from_pytximport=True
            )

    def test_invalid_counts_from_abundance(self, pytximport_adata):
        """Test validation when counts are not raw counts."""
        pytximport_adata.uns["counts_from_abundance"] = "scaledTPM"

        with pytest.raises(ValueError, match="only be used with raw counts"):
            DeseqDataSet(
                adata=pytximport_adata, design="~condition", from_pytximport=True
            )

    def test_length_matrix_shape(self, pytximport_adata):
        """Test validation of length matrix shape."""
        # Create a copy to avoid modifying the fixture
        test_adata = pytximport_adata.copy()

        # Test wrong number of columns (genes) - this should pass AnnData validation
        # but fail our pytximport validation
        test_adata.obsm["length"] = np.random.rand(4, 30)  # Wrong n_vars

        with pytest.raises(ValueError, match="Length matrix shape"):
            DeseqDataSet(adata=test_adata, design="~condition", from_pytximport=True)

    def test_negative_lengths(self, pytximport_adata):
        """Test validation of non-positive gene lengths."""
        # Add some negative lengths
        length_matrix = pytximport_adata.obsm["length"].copy()
        length_matrix[0, 0] = -1.0
        pytximport_adata.obsm["length"] = length_matrix

        with pytest.raises(ValueError, match="non-positive values"):
            DeseqDataSet(
                adata=pytximport_adata, design="~condition", from_pytximport=True
            )


class TestNormalizationFactors:
    """Test normalization factor computation and usage."""

    def test_normalization_factor_computation(self, pytximport_adata):
        """Test that normalization factors are computed correctly."""
        dds = DeseqDataSet(
            adata=pytximport_adata,
            design="~condition",
            from_pytximport=True,
        )
        dds.fit_size_factors()

        # Check normalization factors properties
        norm_factors = dds.obsm["normalization_factors"]
        assert norm_factors.shape == (4, 41127)
        assert np.all(norm_factors > 0)

        # Check that they're different from simple size factors
        # (due to gene-length correction)
        size_factors = dds.obs["size_factors"].values
        norm_factor_means = norm_factors.mean(axis=1)

        # Check that normalization factors vary across genes (not constant)
        norm_factor_std = np.std(norm_factors, axis=1)
        assert np.all(norm_factor_std > 0)  # Should have variation across genes

        # Should have some difference from size factors
        max_diff = np.max(np.abs(norm_factor_means - size_factors))
        assert max_diff > 0.001  # At least some difference

    def test_normalized_counts(self, pytximport_adata):
        """Test that normalized counts are computed correctly."""
        dds = DeseqDataSet(
            adata=pytximport_adata,
            design="~condition",
            from_pytximport=True,
        )
        dds.fit_size_factors()

        # Check normalized counts
        normed_counts = dds.layers["normed_counts"]
        assert normed_counts.shape == (4, 41127)
        assert np.all(normed_counts >= 0)

        # Normalized counts should be different from simple division
        simple_normed = dds.X / dds.obs["size_factors"].values[:, None]
        assert not np.allclose(normed_counts, simple_normed, rtol=0.01)

    def test_comparison_with_standard_deseq2(self, pytximport_adata, standard_adata):
        """Compare pytximport and standard DESeq2 normalization."""
        # pytximport normalization
        dds_pyxtm = DeseqDataSet(
            adata=pytximport_adata,
            design="~condition",
            from_pytximport=True,
        )
        dds_pyxtm.fit_size_factors()

        # Standard normalization
        dds_std = DeseqDataSet(adata=standard_adata, design="~condition")
        dds_std.fit_size_factors()

        # The main difference should be in having normalization factors vs not
        assert "normalization_factors" in dds_pyxtm.obsm
        assert "normalization_factors" not in dds_std.obsm

        # Both should have size factors
        assert "size_factors" in dds_pyxtm.obs
        assert "size_factors" in dds_std.obs

        # Normalization factors should exist and have correct properties
        norm_factors = dds_pyxtm.obsm["normalization_factors"]
        assert norm_factors.shape == (4, 41127)
        assert np.all(norm_factors > 0)


class TestDownstreamAnalysis:
    """Test that downstream analysis works with normalization factors."""

    def test_deseq2_pipeline_with_normalization_factors(self, pytximport_adata):
        """Test full DESeq2 pipeline with pytximport normalization factors."""
        dds = DeseqDataSet(
            adata=pytximport_adata, design="~condition", from_pytximport=True
        )

        # Run full DESeq2 pipeline - may have convergence warnings with small data
        with pytest.warns(
            UserWarning, match="As the residual degrees of freedom is less than 3"
        ):
            dds.deseq2()

        # Check that all expected results are computed
        assert "LFC" in dds.varm
        assert "dispersions" in dds.var
        assert "size_factors" in dds.obs
        assert "normalization_factors" in dds.obsm

        # Test differential expression analysis
        stat_res = DeseqStats(dds, contrast=["condition", "ko", "wt"])
        stat_res.summary()

        # Check results
        assert hasattr(stat_res, "results_df")
        assert "log2FoldChange" in stat_res.results_df.columns
        assert "padj" in stat_res.results_df.columns

        # Results should be reasonable
        results_df = stat_res.results_df
        assert results_df.shape[0] == 41127  # All genes
        assert not results_df["log2FoldChange"].isna().all()

    def test_vst_with_normalization_factors(self, pytximport_adata):
        """Test VST transformation with pytximport normalization factors."""
        dds = DeseqDataSet(
            adata=pytximport_adata, design="~condition", from_pytximport=True
        )

        dds.vst()

        # Check VST results
        assert "vst_counts" in dds.layers
        vst_counts = dds.layers["vst_counts"]
        assert vst_counts.shape == (4, 41127)
        assert np.all(np.isfinite(vst_counts))
