"""
pytximport Integration Example
==============================

This example demonstrates how to use PyDESeq2 with pytximport-derived data
for differential expression analysis with transcript-level quantification data.

pytximport enables the import of transcript-level abundance estimates from
popular RNA-seq quantification tools like Salmon, Kallisto, or RSEM into
PyDESeq2, while accounting for gene length differences between samples
due to differential isoform usage.
"""

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# %%
# Read AnnData object
# -------------------
#
# Make sure that you have rounded the counts in .X to integers, as DESeq2 requires
# integer counts and that pytximport was used with counts_from_abundance=None
# (raw counts) to generate the AnnData object.

adata = ad.read_h5ad("../tests/data/pytximport/test_pytximport.h5ad")

# %%
# Standard usage without pytximport
# ---------------------------------
#
# By default, PyDESeq2 uses standard size factor normalization.

dds_standard = DeseqDataSet(adata=adata, design="~condition")
dds_standard.fit_size_factors()
print(f"Standard normalization: {'normalization_factors' not in dds_standard.obsm}")

# %%
# Explicit pytximport mode
# ------------------------
#
# To use pytximport normalization factors, explicitly set from_pytximport=True.

dds_explicit = DeseqDataSet(adata=adata, design="~condition", from_pytximport=True)
dds_explicit.fit_size_factors()
print(f"Pytximport normalization: {'normalization_factors' in dds_explicit.obsm}")

# %%
# Size factor estimation with normalization factors
# -------------------------------------------------
#
# When pytximport data is used, PyDESeq2 computes normalization factors
# that account for both library size and gene length differences.

dds_explicit.fit_size_factors()

print(f"Size factors computed: {'size_factors' in dds_explicit.obs}")
print(f"Normalization factors computed: {'normalization_factors' in dds_explicit.obsm}")

# Check normalization factors properties
norm_factors = dds_explicit.obsm["normalization_factors"]
size_factors = dds_explicit.obs["size_factors"]

print(f"Normalization factors shape: {norm_factors.shape}")
print(f"Size factors shape: {size_factors.shape}")
print(f"Size factors: {size_factors.values}")
print(f"Norm factors range: {norm_factors.min():.3f} - {norm_factors.max():.3f}")

# %%
# Plot distribution of normalization factors
# ------------------------------------------
#
# Normalization factors should vary around 1, reflecting gene length
# differences between samples.
plt.figure(figsize=(8, 5))
plt.hist(
    norm_factors.flatten(),
    bins=list(np.arange(0.5, 1.6, 0.1).astype(float)),
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
)
plt.axvline(x=1, color="red", linestyle="--", linewidth=2, label="x = 1")
plt.title("Distribution of Normalization Factors")
plt.xlabel("Normalization Factor")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Full differential expression analysis
# -------------------------------------
#
# The complete DESeq2 pipeline works seamlessly with pytximport data.

# Run the full DESeq2 pipeline
dds_explicit.deseq2()

# Perform statistical testing
stat_res = DeseqStats(dds_explicit, contrast=["condition", "ko", "wt"])
stat_res.summary()
results_df = stat_res.results_df

print("Differential expression results:")
print(f"Total genes analyzed: {len(results_df)}")
print(f"Significant genes (padj < 0.05): {sum(results_df['padj'] < 0.05)}")

# Display top results
print("\nTop 5 most significant genes:")
top_results = results_df.sort_values("padj").head()
print(top_results[["baseMean", "log2FoldChange", "pvalue", "padj"]])
stat_res.plot_MA()

# %%
# Differential expression analysis without normalization factors
# --------------------------------------------------------------
#
# Compare results with standard DESeq2 normalization (without length correction).

# Read standard AnnData (generated with counts_from_abundance="length_scaled_tpm")
adata_standard = ad.read_h5ad(
    "../tests/data/pytximport/test_pytximport_length_scaled.h5ad"
)

dds_standard = DeseqDataSet(adata=adata_standard, design="~condition")
dds_standard.deseq2()

stat_res_standard = DeseqStats(dds_standard, contrast=["condition", "ko", "wt"])
stat_res_standard.summary()
results_standard = stat_res_standard.results_df
print("\nTop 5 most significant genes:")
top_results = results_standard.sort_values("padj").head()
print(top_results[["baseMean", "log2FoldChange", "pvalue", "padj"]])
stat_res_standard.plot_MA()

# %%
# Comparison with standard normalization
# --------------------------------------
#
# Compare results with standard DESeq2 normalization (without length correction).
pytximport_sig = sum(results_df["padj"] < 0.05)
standard_sig = sum(results_standard["padj"] < 0.05)

print("\nComparison of significant genes (padj < 0.05):")
print(f"pytximport normalization: {pytximport_sig}")
print(f"Standard normalization: {standard_sig}")

# %%
# Variance Stabilizing Transformation (VST)
# -----------------------------------------
#
# VST also works with pytximport normalization factors.

dds_vst = DeseqDataSet(adata=adata, design="~condition", from_pytximport=True)
dds_vst.vst()

vst_counts = dds_vst.layers["vst_counts"]
print("\nVST transformation completed")
print(f"VST counts shape: {vst_counts.shape}")
print(f"VST counts range: {vst_counts.min():.3f} - {vst_counts.max():.3f}")
