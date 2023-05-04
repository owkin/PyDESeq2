from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd


def deseq2_norm(
    counts: Union[pd.DataFrame, np.ndarray]
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
    """
    Return normalized counts and size_factors.

    Uses the median of ratios method.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
            Raw counts. One column per gene, one row per sample.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
        DESeq2 normalization factors.
    """

    # Compute gene-wise mean log counts
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
    logmeans = log_counts.mean(0)
    # Filter out genes with -∞ log means
    filtered_genes = ~np.isinf(logmeans)
    # Subtract filtered log means from log counts
    if isinstance(log_counts, pd.DataFrame):
        log_ratios = log_counts.loc[:, filtered_genes] - logmeans[filtered_genes]
    else:
        log_ratios = log_counts[:, filtered_genes] - logmeans[filtered_genes]
    # Compute sample-wise median of log ratios
    log_medians = np.median(log_ratios, axis=1)
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts / size_factors[:, None]
    return deseq2_counts, size_factors
