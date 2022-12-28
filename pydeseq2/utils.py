import multiprocessing
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.special import polygamma
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import pydeseq2
from pydeseq2.grid_search import grid_fit_alpha
from pydeseq2.grid_search import grid_fit_beta
from pydeseq2.grid_search import grid_fit_shrink_beta


def load_data(
    modality="raw_counts",
    cancer_type="synthetic",
    debug=False,
    debug_seed=42,
):
    """Load synthetic or TCGA data (gene raw counts or clinical) for a given cancer type.

    May load either clinical or rna-seq data.The synthetic data is part of this
    repo, but TCGA data should be downloaded as per the instructions in `datasets/`.

    Parameters
    ----------
    modality : str
        Data modality. "raw_counts" or "clinical".

    cancer_type : str
        The cancer type for which to return gene expression data.
        If "synthetic", will return the synthetic data that is used for CI unit tests.
        Otherwise, must be a valid TCGA dataset. (default: "synthetic").

    debug : bool
        If true, subsample 10 samples and 100 genes at random.
        Only supported in "pooled" mode for now. (default: False).

    debug_seed : int
        Seed for the debug mode. (default: 42).

    Returns
    -------
    pandas.DataFrame
        Requested data modality.
    """

    assert modality in ["raw_counts", "clinical"], (
        "The modality argument must be one of the following: " "raw_counts, clinical"
    )

    assert cancer_type in [
        "synthetic",
        "TCGA-BRCA",
        "TCGA-COAD",
        "TCGA-LUAD",
        "TCGA-LUSC",
        "TCGA-PAAD",
        "TCGA-PRAD",
        "TCGA-READ",
        "TCGA-SKCM",
    ], (
        "The cancer_type argument must be one of the following: "
        "syntetic, TCGA-BRCA, TCGA-COAD, TCGA-LUAD, TCGA-LUSC, "
        "TCGA-PAAD, TCGA-PRAD, TCGA-READ, TCGA-SKCM"
    )
    # Load data
    if cancer_type == "synthetic":
        datasets_path = Path(pydeseq2.__file__).parent.parent / "tests"
        path_to_data = datasets_path / "data"

        if modality == "raw_counts":
            df = pd.read_csv(
                path_to_data / "test_counts.csv",
                sep=",",
                index_col=0,
            ).T
        elif modality == "clinical":
            df = pd.read_csv(
                path_to_data / "test_clinical.csv",
                sep=",",
                index_col=0,
            )
    else:
        datasets_path = Path(pydeseq2.__file__).parent.parent / "datasets"
        path_to_data = datasets_path / "tcga_data"

        if modality == "raw_counts":
            df = pd.read_csv(
                path_to_data / "Gene_expressions" / f"{cancer_type}_raw_RNAseq.tsv.gz",
                compression="gzip",
                sep="\t",
                index_col=0,
            ).T
        elif modality == "clinical":
            df = pd.read_csv(
                path_to_data / "Clinical" / f"{cancer_type}_clinical.tsv.gz",
                compression="gzip",
                sep="\t",
                index_col=0,
            )

    if debug:
        # subsample 10 samples and 100 genes
        df = df.sample(n=10, axis=0, random_state=debug_seed)
        if modality == "raw_counts":
            df = df.sample(n=100, axis=1, random_state=debug_seed)

    return df


def test_valid_counts(counts_df):
    """Test that the count matrix contains valid inputs.

    More precisely, test that inputs are non-negative integers.

    Parameters
    ----------
    counts_df : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.
    """
    if counts_df.isna().any().any():
        raise ValueError("NaNs are not allowed in the count matrix.")
    if ~counts_df.apply(
        lambda s: pd.to_numeric(s, errors="coerce").notnull().all()
    ).all():
        raise ValueError("The count matrix should only contain numbers.")
    if (counts_df % 1 != 0).any().any():
        raise ValueError("The count matrix should only contain integers.")
    if (counts_df < 0).any().any():
        raise ValueError("The count matrix should only contain non-negative values.")


def build_design_matrix(
    clinical_df, design="high_grade", ref=None, expanded=False, intercept=True
):
    """Build design_matrix matrix for DEA.

    Only single factor, 2-level designs are currently supported.
    Unless specified, the reference factor is chosen alphabetically.

    Parameters
    ----------
    clinical_df : pandas.DataFrame
        DataFrame containing clinical information.
        Must be indexed by sample barcodes, and contain a "high_grade" column.

    design : str
        Name of the column of clinical_df to be used as a design_matrix variable.
        (default: "high_grade").

    ref : str
        The factor to use as a reference. Must be one of the values taken by the design.
        If None, the reference will be chosen alphabetically (last in order).
        (default: None).

    expanded : bool
        If true, use one column per category. Else, use a single column.
        (default: False).

    intercept : bool
        If true, add an intercept (a column containing only ones). (default: True).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes.
    """

    design_matrix = pd.get_dummies(clinical_df[design])

    if design_matrix.shape[-1] == 1:
        raise ValueError(
            f"The design factor takes only one value: "
            f"{design_matrix.columns.values.tolist()}, but two are necessary."
        )
    elif design_matrix.shape[-1] > 2:
        raise ValueError(
            f"The design factor takes more than two values: "
            f"{design_matrix.columns.values.tolist()}, but should have exactly two."
        )
    # Sort columns alphabetically
    design_matrix = design_matrix.reindex(
        sorted(design_matrix.columns, reverse=True), axis=1
    )
    if ref is not None:  # Put reference level last
        try:
            ref_level = design_matrix.pop(ref)
        except KeyError as e:
            print(
                "The reference level must correspond to"
                " one of the design factor values."
            )
            raise e
        design_matrix.insert(1, ref, ref_level)
    if not expanded:  # drop last factor
        design_matrix.drop(columns=design_matrix.columns[-1], axis=1, inplace=True)
    if intercept:
        design_matrix.insert(0, "intercept", 1.0)
    return design_matrix


def dispersion_trend(normed_mean, coeffs):
    r"""Return dispersion trend from normalized counts.

     :math:`a_1/ \mu + a_0`.

    Parameters
    ----------
    normed_mean : float or ndarray
        Mean of normalized counts for a given gene or set of genes.

    coeffs : ndarray or pd.Series
        Fitted dispersion trend coefficients :math:`a_0` and :math:`a_1`.

    Returns
    -------
    float
        Dispersion trend :math:`a_1/ \mu + a_0`.
    """
    if type(coeffs) == pd.Series:
        return coeffs["a0"] + coeffs["a1"] / normed_mean
    else:
        return coeffs[0] + coeffs[1] / normed_mean


def nb_nll(y, mu, alpha):
    """Negative log-likelihood of a negative binomial.

    Unvectorized version.

    Parameters
    ----------
    y : ndarray
        Observations.

    mu : float
        Mean of the distribution.

    alpha : float
        Dispersion of the distribution,
        s.t. the variance is :math:`\\mu + \\alpha * \\mu^2`.

    Returns
    -------
    float
        Negative log likelihood of the observations y
        following :math:`NB(\\mu, \\alpha)`.
    """

    n = len(y)
    alpha_neg1 = 1 / alpha
    logbinom = gammaln(y + alpha_neg1) - gammaln(y + 1) - gammaln(alpha_neg1)
    return (
        n * alpha_neg1 * np.log(alpha)
        + (-logbinom + (y + alpha_neg1) * np.log(alpha_neg1 + mu) - y * np.log(mu)).sum()
    )


def dnb_nll(y, mu, alpha):
    """Gradient of the negative log-likelihood of a negative binomial.

    Unvectorized.

    Parameters
    ----------
    y : ndarray
        Observations.

    mu : float
        Mean of the distribution.

    alpha : float
        Dispersion of the distribution,
        s.t. the variance is :math:`\\mu + \\alpha * \\mu^2`.

    Returns
    -------
    float
        Derivative of negative log likelihood of NB w.r.t. :math:`\\alpha`.
    """

    alpha_neg1 = 1 / alpha
    ll_part = (
        alpha_neg1**2
        * (
            polygamma(0, alpha_neg1)
            - polygamma(0, y + alpha_neg1)
            + np.log(1 + mu * alpha)
            + (y - mu) / (mu + alpha_neg1)
        ).sum()
    )

    return -ll_part


def irls_solver(
    counts,
    size_factors,
    design_matrix,
    disp,
    min_mu=0.5,
    beta_tol=1e-8,
    min_beta=-30,
    max_beta=30,
    optimizer="L-BFGS-B",
):
    r"""Fit a NB GLM wit log-link to predict counts from the design matrix.

    See equations (1-2) in the DESeq2 paper.
    Uses the L-BFGS-B, more stable than Fisher scoring.

    Parameters
    ----------
    counts : pandas.Series
        Raw counts for a given gene.

    size_factors : pandas.Series
        Sample-wise scaling factors (obtained from median-of-ratios).

    design_matrix : pandas.DataFrame
        Design matrix.

    disp : pandas.Series
        Gene-wise dispersion priors.

    min_mu : float
        Lower bound on estimated means, to ensure numerical stability. (default: 0.5).

    beta_tol : float
        Stopping criterion for IRWLS:
        :math:`abs(dev - old_dev) / (abs(dev) + 0.1) < beta_tol`. (default: 1e-8).

    min_beta : int
        Lower-bound on LFC. (default: -30).

    max_beta : int
        Upper-bound on LFC. (default: -30).

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'BFGS' or 'L-BFGS-B'.
        NB: only 'L-BFGS-B' ensures that LFCS will
        lay in the [min_beta, max_beta] range. (default: 'BFGS').

    Returns
    -------
    beta: ndarray
        Fitted (basemean, lfc) coefficients of negative binomial GLM.

    mu: ndarray
        Means estimated from size factors and beta:
        :math:`\\mu = s_{ij} \\exp(\\beta^t X)`.

    H: ndarray
        Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}` covariance matrix.
    """

    assert optimizer in ["BFGS", "L-BFGS-B"]

    p = design_matrix.shape[1]

    # converting to numpy for speed
    if type(counts) == pd.Series:
        counts = counts.values
    if type(size_factors) == pd.Series:
        size_factors = size_factors.values
    if type(design_matrix) == pd.Series:
        X = design_matrix.values
    else:
        X = design_matrix

    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(X) == p:
        Q, R = np.linalg.qr(X)
        y = np.log(counts / size_factors + 0.1)
        beta_init = solve(R, Q.T @ y)
        beta = beta_init

    dev = 1000
    dev_ratio = 1.0

    D = np.diag(np.repeat(1e-6, p))
    mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)

    converged = True
    while dev_ratio > beta_tol:

        W = mu / (1.0 + mu * disp)
        z = np.log(mu / size_factors) + (counts - mu) / mu
        H = (X.T * W) @ X + D
        beta_hat = solve(H, X.T @ (W * z), assume_a="pos")
        if sum(np.abs(beta_hat) > max_beta) > 0:
            # If IRLS starts diverging, use L-BFGS-B
            def f(beta):
                # closure to minimize
                mu_ = np.maximum(size_factors * np.exp(X @ beta), min_mu)
                return nb_nll(counts, mu_, disp) + 0.5 * (D @ beta**2).sum()

            def df(beta):
                mu_ = np.maximum(size_factors * np.exp(X @ beta), min_mu)
                return (
                    -X.T @ counts
                    + ((1 / disp + counts) * mu_ / (1 / disp + mu_)) @ X
                    + D @ beta
                )

            res = minimize(
                f,
                beta_init,
                jac=df,
                method=optimizer,
                bounds=[(min_beta, max_beta), (min_beta, max_beta)]
                if optimizer == "L-BFGS-B"
                else None,
            )

            beta = res.x
            mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
            converged = res.success

            if not res.success:
                beta = grid_fit_beta(
                    counts,
                    size_factors,
                    X,
                    disp,
                )
                mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
            break

        beta = beta_hat
        mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
        # Compute deviation
        old_dev = dev
        # Replaced deviation with -2 * nll, as in the R code
        dev = -2 * nb_nll(counts, mu, disp)
        dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)

    # Compute H diagonal (useful for Cook distance outlier filtering)
    W = mu / (1.0 + mu * disp)
    W_sq = np.sqrt(W)
    XtWX = (X.T * W) @ X + D
    H = W_sq * np.diag(X @ np.linalg.inv(XtWX) @ X.T) * W_sq
    # Return an UNthresholded mu (as in the R code)
    # Previous quantities are estimated with a threshold though
    mu = size_factors * np.exp(X @ beta)
    return beta, mu, H, converged


def fit_alpha_mle(
    y,
    X,
    mu,
    alpha_hat,
    min_disp,
    max_disp,
    prior_disp_var=None,
    cr_reg=True,
    prior_reg=False,
    optimizer="L-BFGS-B",
):
    """Estimate the dispersion parameter of a negative binomial GLM.

    Note: it is possible to pass pandas Series as y, X and mu arguments but using numpy
    arrays makes the code significantly faster.

    Parameters
    ----------
    y : ndarray
        Raw counts for a given gene.

    X : ndarray
        Design matrix.

    mu : ndarray
        Mean estimation for the NB model.

    alpha_hat : float
        Initial dispersion estimate.

    min_disp : float
        Lower threshold for dispersion parameters.

    max_disp : float
        Upper threshold for dispersion parameters.

    prior_disp_var : float
        Prior dispersion variance.

    cr_reg : bool
        Whether to use Cox-Reid regularization. (default: True).

    prior_reg : bool
        Whether to use prior log-residual regularization. (default: False).

    optimizer : str
        Optimizing method to use. Accepted values: 'BFGS' or 'L-BFGS-B'.
        (default: 'BFGS').

    Returns
    -------
    float
        Dispersion estimate.

    bool
        Whether L-BFGS-B converged. If not, dispersion is estimated using grid search.
    """

    assert optimizer in ["BFGS", "L-BFGS-B"]

    if prior_reg:
        assert (
            prior_disp_var is not None
        ), "Sigma_prior is required for prior regularization"

    def loss(log_alpha):
        # closure to be minimized
        alpha = np.exp(log_alpha)
        W = mu / (1 + mu * alpha)
        reg = 0
        if cr_reg:
            reg += 0.5 * np.linalg.slogdet((X.T * W) @ X)[1]
        if prior_reg:
            reg += (np.log(alpha) - np.log(alpha_hat)) ** 2 / (2 * prior_disp_var)
        return nb_nll(y, mu, alpha) + reg

    def dloss(log_alpha):
        # gradient closure
        alpha = np.exp(log_alpha)
        W = mu / (1 + mu * alpha)
        dW = -(W**2)
        reg_grad = 0
        if cr_reg:
            reg_grad += 0.5 * (np.linalg.inv((X.T * W) @ X) * ((X.T * dW) @ X)).sum()
        if prior_reg:
            reg_grad += (np.log(alpha) - np.log(alpha_hat)) / (alpha * prior_disp_var)
        return dnb_nll(y, mu, alpha) + reg_grad

    res = minimize(
        lambda x: loss(x[0]),
        x0=[np.log(alpha_hat)],
        jac=dloss,
        method=optimizer,
        bounds=[(np.log(min_disp), np.log(max_disp))]
        if optimizer == "L-BFGS-B"
        else None,
    )

    if res.success:
        return np.exp(res.x[0]), res.success
    else:
        return (
            np.exp(grid_fit_alpha(y, X, mu, alpha_hat, min_disp, max_disp)),
            res.success,
        )


def trimmed_mean(x, trim=0.1, **kwargs):
    """Return trimmed mean.

    Compute the mean after trimming data of its smallest and largest quantiles.

    Parameters
    ----------
    x : ndarray
        Data whose mean to compute.

    trim : float
        Fraction of data to trim at each end. (default: 0.1).

    **kwargs
        Keyword arguments, useful to pass axis.

    Returns
    -------
    float or ndarray :
        Trimmed mean.
    """

    assert trim <= 0.5
    if "axis" in kwargs:
        axis = kwargs.get("axis")
        s = np.sort(x, axis=axis)
        n = x.shape[axis]
        ntrim = floor(n * trim)
        return np.take(s, np.arange(ntrim, n - ntrim), axis).mean(axis)
    else:
        n = len(x)
        s = np.sort(x)
        ntrim = floor(n * trim)
        return s[ntrim : n - ntrim].mean()


def trimmed_cell_variance(counts, cells):
    """Return trimmed variance of counts according to condition.

    Compute the variance after trimming data of its smallest and largest elements,
    grouped by cohorts, and return the max across cohorts.
    The trim factor is a function of data size.

    Parameters
    ----------
    counts : pandas.DataFrame
        Sample-wise gene counts.

    cells : pandas.DataFrame
        Cohort affiliation of each sample.

    Returns
    -------
    pandas.Series :
        Gene-wise trimmed variance estimate.
    """

    # how much to trim at different n
    trimratio = (1 / 3, 1 / 4, 1 / 8)
    # returns an index for the vector above for three sample size bins

    def trimfn(x):
        return 2 if x >= 23.5 else 1 if x >= 3.5 else 0

    ns = cells.value_counts()
    cell_means = pd.DataFrame(index=counts.columns)
    for lvl in cells.unique():
        cell_means[lvl] = trimmed_mean(
            counts.loc[cells == lvl], trim=trimratio[trimfn(ns[lvl])], axis=0
        )
    qmat = cell_means[cells].T
    qmat.index = cells.index
    sqerror = (counts - qmat) ** 2

    varEst = pd.DataFrame(index=counts.columns)
    for lvl in cells.unique():
        scale = [2.04, 1.86, 1.51][trimfn(ns[lvl])]
        varEst[lvl] = scale * trimmed_mean(
            sqerror[cells == lvl], trim=trimratio[trimfn(ns[lvl])], axis=0
        )

    return varEst.max(axis=1)


def trimmed_variance(x, trim=0.125, axis=1):
    """Return trimmed variance.

     Compute the variance after trimming data of its smallest and largest quantiles.

    Parameters
    ----------
    x : ndarray
        Data whose trimmed variance to compute.

    trim : float
        Fraction of data to trim at each end. (default: 0.1).

    axis : int
        Dimension along which to compute variance.

    Returns
    -------
    ndarray
        Trimmed variances.
    """

    rm = trimmed_mean(x, trim=trim, axis=axis)
    sqerror = (x - rm) ** 2
    # scale due to trimming of large squares
    return 1.51 * trimmed_mean(sqerror, trim=trim, axis=axis)


def fit_lin_mu(y, size_factors, X, min_mu=0.5):
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    y : ndarray
        Raw counts for a given gene.

    size_factors : ndarray
        Sample-wise scaling factors (obtained from median-of-ratios).

    X : ndarray
        Design matrix.

    min_mu : float
        Lower threshold for fitted means, for numerical stability. (default: 0.5).

    Returns
    -------
    float
        Estimated mean.
    """

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y / size_factors)
    mu_hat = size_factors * reg.predict(X)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return np.maximum(mu_hat, min_mu)


def wald_test(X, disp, lfc, mu, D, idx=-1):
    """Run Wald test for differential expression.

    Computes Wald statistics, standard error and p-values from
    dispersion and LFC estimates.

    Parameters
    ----------
    X : ndarray
        Design matrix.

    disp : float
        Dispersion estimate.

    lfc : float
        Log-fold change estimate (in natural log scale).

    mu : float
        Mean estimation for the NB model.

    D : ndarray
        Regularization factors.

    idx : int
        Index of design factor (in design matrix). (default: -1).

    Returns
    -------
    wald_p_value : float
        Estimated p-value.

    wald_statistic : float
        Wald statistic.

    wald_se : float
        Standard error of the Wald statistic.
    """

    # Build covariance matrix estimator
    W = np.diag(mu / (1 + mu * disp))
    M = X.T @ W @ X
    H = np.linalg.inv(M + D)
    S = H @ M @ H
    # Evaluate standard error and Wald statistic
    c = np.zeros(lfc.shape)
    c[idx] = 1
    wald_se = np.sqrt(c.T @ S @ c)
    wald_statistic = c.T @ lfc / wald_se
    wald_p_value = 2 * norm.sf(np.abs(wald_statistic))
    return wald_p_value, wald_statistic, wald_se


def fit_rough_dispersions(counts, size_factors, design_matrix):
    """ "Rough dispersion" estimates from linear model, as per the R code.

    Used as initial estimates in DeseqDataSet._fit_MoM_dispersions.

    Parameters
    ----------
    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.Series
        DESeq2 normalization factors.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    pandas.Series
        Estimated dispersion parameter for each gene.
    """

    m, p = design_matrix.shape
    normed_counts = counts.div(size_factors, 0)
    # Exclude genes with all zeroes
    normed_counts = normed_counts.loc[:, ~(normed_counts == 0).all()]
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, normed_counts)
    y_hat = reg.predict(design_matrix)
    y_hat = np.maximum(y_hat, 1)
    alpha_rde = (((normed_counts - y_hat) ** 2 - y_hat) / ((m - p) * y_hat**2)).sum(0)
    return np.maximum(alpha_rde, 0)


def fit_moments_dispersions(counts, size_factors):
    """Dispersion estimates based on moments, as per the R code.

    Used as initial estimates in DeseqDataSet._fit_MoM_dispersions.

    Parameters
    ----------
    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.Series
        DESeq2 normalization factors.

    Returns
    -------
    pandas.Series
        Estimated dispersion parameter for each gene.
    """
    normed_counts = counts.div(size_factors, 0)
    # Exclude genes with all zeroes
    normed_counts = normed_counts.loc[:, ~(normed_counts == 0).all()]
    # mean inverse size factor
    s_mean_inv = (1 / size_factors).mean()
    mu = normed_counts.mean(0)
    sigma = normed_counts.var(0, ddof=1)
    # ddof=1 is to use an unbiased estimator, as in R
    # NaN (variance = 0) are replaced with 0s
    return ((sigma - s_mean_inv * mu) / mu**2).fillna(0)


def robust_method_of_moments_disp(normed_counts, design_matrix):
    """Perform dispersion estimation using a method of trimmed moments.

    Used for outlier detection based on Cook's distance.

    Parameters
    ----------
    normed_counts : pandas.DataFrame
        DF of deseq2-normalized read counts. Rows: samples, columns: genes.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    pandas.Series
        Trimmed method of moment dispersion estimates.
        Used for outlier detection based on Cook's distance.
    """

    # if there are 3 or more replicates in any cell
    three_or_more = design_matrix[design_matrix.columns[-1]].value_counts() >= 3
    if three_or_more.any():
        v = trimmed_cell_variance(normed_counts, design_matrix.iloc[:, -1].astype(int))
        # last argument is non-intercept.
    else:
        v = trimmed_variance(normed_counts)
    m = normed_counts.mean(0)
    alpha = (v - m) / m**2
    # cannot use the typical min_disp = 1e-8 here or else all counts in the same
    # group as the outlier count will get an extreme Cook's distance
    minDisp = 0.04
    alpha = np.maximum(alpha, minDisp)
    return alpha


def get_num_processes(n_cpus=None):
    """Return the number of processes to use for multiprocessing.

    Returns the maximum number of available cpus by default.

    Parameters
    ----------
    n_cpus : int
        Desired number of cpus. If None, will return the number of available cpus.
        (default: None).

    Returns
    -------
    int
        Number of processes to spawn.
    """

    if n_cpus is None:
        try:
            n_processes = multiprocessing.cpu_count()
        except NotImplementedError:
            n_processes = 5  # arbitrary default
    else:
        n_processes = n_cpus

    return n_processes


def nbinomGLM(
    design_matrix,
    counts,
    size,
    offset,
    prior_no_shrink_scale,
    prior_scale,
    optimizer="L-BFGS-B",
    shrink_index=1,
):
    """Fit a negative binomial MAP LFC using an apeGLM prior.

    Only the LFC is shrinked, and not the intercept.

    Parameters
    ----------
    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the LFC parameter.

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'. (default: 'Newton-CG').

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: 1).

    Returns
    -------
    beta: ndarray
        2-element array, containing the intercept (first) and the LFC (second).

    inv_hessian: ndarray
        Inverse of the Hessian of the objective at the estimated MAP LFC.

    converged: bool
        Whether L-BFGS-B converged.
    """
    beta_init = np.array([0.1, -0.1])
    no_shrink_index = 1 - shrink_index

    # Set optimization scale
    scale_cnst = nbinomFn(
        np.array([0, 0]),
        design_matrix,
        counts,
        size,
        offset,
        prior_no_shrink_scale,
        prior_scale,
    )
    scale_cnst = np.maximum(scale_cnst, 1)

    def f(beta, cnst=scale_cnst):
        # Function to optimize
        return (
            nbinomFn(
                beta,
                design_matrix,
                counts,
                size,
                offset,
                prior_no_shrink_scale,
                prior_scale,
            )
            / cnst
        )

    def df(beta, cnst=scale_cnst):
        # Gradient of the function to optimize
        xbeta = design_matrix @ beta
        d_neg_prior = np.array(
            [
                beta[no_shrink_index] / prior_no_shrink_scale**2,
                2 * beta[shrink_index] / (prior_scale**2 + beta[shrink_index] ** 2),
            ]
        )

        d_nll = (
            counts - (counts + size) / (1 + size * np.exp(-xbeta - offset))
        ) @ design_matrix

        return (d_neg_prior - d_nll) / cnst

    def ddf(beta, cnst=scale_cnst):
        # Hessian of the function to optimize
        xbeta = design_matrix @ beta
        exp_xbeta_off = np.exp(xbeta + offset)
        frac = (counts + size) * size * exp_xbeta_off / (size + exp_xbeta_off) ** 2
        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta[shrink_index] ** 2)
            / (prior_scale**2 + beta[shrink_index] ** 2) ** 2
        )
        return (
            1 / cnst * ((design_matrix.T * frac) @ design_matrix + np.diag([h11, h22]))
        )

    res = minimize(
        f,
        beta_init,
        jac=df,
        hess=ddf if optimizer == "Newton-CG" else None,
        method=optimizer,
    )

    beta = res.x
    converged = res.success

    if not converged:
        # If the solver failed, fit using grid search (slow)
        beta = grid_fit_shrink_beta(
            counts,
            offset,
            design_matrix,
            size,
            prior_no_shrink_scale,
            prior_scale,
            scale_cnst,
            grid_length=60,
            min_beta=-30,
            max_beta=30,
        )

    inv_hessian = np.linalg.inv(ddf(beta, 1))

    return beta, inv_hessian, converged


def nbinomFn(
    beta,
    design_matrix,
    counts,
    size,
    offset,
    prior_no_shrink_scale,
    prior_scale,
    shrink_index=1,
):
    """Return the NB negative likelihood with apeGLM prior.

    Use for LFC shrinkage.

    Parameters
    ----------
    beta : ndarray
        2-element array: intercept and LFC coefficients.

    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the intercept.

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: 1).

    Returns
    -------
    float
        Sum of the NB negative likelihood and apeGLM prior.
    """

    no_shrink_index = 1 - shrink_index
    xbeta = design_matrix @ beta
    prior = beta[no_shrink_index] ** 2 / (2 * prior_no_shrink_scale**2) + np.log1p(
        (beta[shrink_index] / prior_scale) ** 2
    )

    nll = (
        counts * xbeta - (counts + size) * np.logaddexp(xbeta + offset, np.log(size))
    ).sum(0)

    return prior - nll
