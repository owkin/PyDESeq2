"""
Pydeseq2 with real data
======================================================

This experiment aims to run pydeseq2 on real data.

"""

import matplotlib.pyplot as plt
import numpy as np

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

# %%
# Data loading
# ------------
#
# Let's first download and load the real data.

counts_ori = load_example_data(
    modality="raw_counts",
    dataset="real",
    debug=False,
)

metadata = load_example_data(
    modality="metadata",
    dataset="real",
    debug=False,
)

print(counts_ori.tail())

# %%
# In this example, the counts data contains both EnsemblIDs
# and gene symbols (gene names).
# We also see that some EnsemblID do not map to any gene symbol.
# We decide to stay with EnsemblID rather than gene symbol and
# continue with some processing step.

# %%
# Data filtering read counts
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

counts_df = counts_ori.drop(columns="Gene Name")
# remove the gene name for now
counts_df = counts_df.set_index("Gene ID").T
# now we have a matrice of shape n_samples x n_genes

# %%
# Further processing might be needed because some genes
# have very high counts on average.

# %%
print(metadata)

# %%
# Data filtering metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

metadata = metadata.set_index("Run")[["Sample Characteristic[biopsy site]"]].rename(
    columns={"Sample Characteristic[biopsy site]": "condition"}
)

# %%
# Now that we have loaded and filtered our data, we may proceed with the differential
# analysis.


# %%
# Single factor analysis
# --------------------------
#
# In this analysis, we use the ``condition``
# column as our design factor. That is, we compare gene expressions of samples that have
# ``primary tumor`` to those that have ``normal`` condition.
#

dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design_factors="condition",
    refit_cooks=True,
    n_cpus=8,
)

dds.deseq2()

# %%
# Generate summary statistics
# --------------------------
# We are here interested in comparing "primary tumor vs "normal" samples

stat_res_tumor_vs_normal = DeseqStats(
    dds, contrast=["condition", "primary tumor", "normal"], n_cpus=8
)

# As we did not really performed a thorough exploration of the data
# and gene counts, we would like to test for a lfc of 1 at least

stat_res_tumor_vs_normal.summary(lfc_null=1, alt_hypothesis="greaterAbs")

# Some visualization

stat_res_tumor_vs_normal.plot_MA(s=20)

# Volcano plot

plt.scatter(
    stat_res_tumor_vs_normal.results_df["log2FoldChange"],
    -np.log(stat_res_tumor_vs_normal.results_df["padj"]),
)
