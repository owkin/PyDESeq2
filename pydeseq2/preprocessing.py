import numpy as np


def deseq2_norm(counts):
    """
    Return normalized counts and size_factors.

    Uses the median of ratios method.

    Parameters
    ----------
    counts : pandas.DataFrame
            Raw counts. One column per gene, rows are indexed by sample barcodes.

    Returns
    -------
    deseq2_counts : pandas.DataFrame
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame
        DESeq2 normalization factors.
    """
    # Compute gene-wise mean log counts
    log_counts = counts.apply(np.log)
    logmeans = log_counts.mean(0)
    # Filter out genes with -∞ log means
    filtered_genes = ~np.isinf(logmeans).values
    # Subtract filtered log means from log counts
    log_ratios = log_counts.iloc[:, filtered_genes] - logmeans[filtered_genes]
    # Compute sample-wise median of log ratios
    log_medians = log_ratios.median(1)
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts.div(size_factors, 0)
    return deseq2_counts, size_factors
