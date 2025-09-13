import numpy as np
import pandas as pd


def deseq2_norm(
    counts: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Return normalized counts and size_factors.

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
    logmeans, filtered_genes = deseq2_norm_fit(counts)
    deseq2_counts, size_factors = deseq2_norm_transform(counts, logmeans, filtered_genes)
    return deseq2_counts, size_factors


def deseq2_norm_fit(counts: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``logmeans`` and ``filtered_genes``, needed in the median of ratios method.

    ``Logmeans`` and ``filtered_genes`` can then be used to normalize external datasets.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
            Raw counts. One column per gene, one row per sample.

    Returns
    -------
    logmeans : ndarray
        Gene-wise mean log counts.

    filtered_genes : ndarray
        Genes whose log means are different from -∞.
    """
    # Compute gene-wise mean log counts
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
    logmeans = log_counts.mean(0)
    # Filter out genes with -∞ log means
    filtered_genes = ~np.isinf(logmeans)

    return logmeans, filtered_genes


def deseq2_norm_transform(
    counts: pd.DataFrame | np.ndarray,
    logmeans: np.ndarray,
    filtered_genes: np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Return normalized counts and size factors from the median of ratios method.

    Can be applied on external dataset, using the ``logmeans`` and ``filtered_genes``
    previously computed in the ``fit`` function.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
            Raw counts. One column per gene, one row per sample.

    logmeans : ndarray
        Gene-wise mean log counts.

    filtered_genes : ndarray
        Genes whose log means are different from -∞.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
        DESeq2 normalization factors.
    """
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
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


def estimate_norm_factors(
    counts: pd.DataFrame | np.ndarray,
    norm_matrix: np.ndarray,
    control_mask: np.ndarray | None = None,
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Estimate normalization factors from counts and normalization matrix.

    Implements the DESeq2 estimateNormFactors algorithm for pytximport integration.
    This function computes normalization factors that account for both library size
    and gene length differences.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
        Raw counts. One column per gene, one row per sample.

    norm_matrix : ndarray
        Normalization matrix. Should have the same dimensions as counts.

    control_mask : ndarray, optional
        Boolean mask indicating which genes to use as control genes for size
        factor estimation. If None, all genes are used.

    Returns
    -------
    counts_adj : pandas.DataFrame or ndarray
        Counts adjusted by the normalization matrix.

    normed_counts : pandas.DataFrame or ndarray
        Normalized counts using the computed normalization factors.

    norm_factors : ndarray
        Computed normalization factors that account for both library size
        and gene length differences.

    logmeans : ndarray
        Gene-wise mean log counts from the adjusted counts.

    filtered_genes : ndarray
        Genes whose log means are different from -∞ in the adjusted counts.
    """
    # Divide counts by normalization matrix
    counts_adj = counts / norm_matrix

    # Compute size factors from adjusted counts using standard DESeq2 method
    logmeans, filtered_genes = deseq2_norm_fit(counts_adj)

    # Apply control gene mask if provided
    if control_mask is not None:
        filtered_genes = filtered_genes & control_mask

    _, size_factors = deseq2_norm_transform(counts_adj, logmeans, filtered_genes)

    # Compute normalization factors: (norm_matrix.T * size_factors).T
    # Broadcasting: norm_matrix (samples x genes) * size_factors (samples,)
    norm_factors = norm_matrix * size_factors[:, None]

    # Normalize to geometric mean of 1 (row-wise geometric mean)
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_norm_factors = np.log(norm_factors)
        log_geom_means = np.mean(log_norm_factors, axis=1, keepdims=True)
        norm_factors = norm_factors / np.exp(log_geom_means)

    # Compute normalized counts
    normed_counts = counts / norm_factors

    # TODO: Check whether we actually have to calculate logmeans and filtered_genes again
    # from the normed counts, or if we can just return the previous ones.
    logmeans, filtered_genes = deseq2_norm_fit(normed_counts)

    return counts_adj, normed_counts, norm_factors, logmeans, filtered_genes
