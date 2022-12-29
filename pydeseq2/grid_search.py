import numpy as np
from scipy.special import gammaln

import pydeseq2.utils


def vec_nb_nll(y, mu, alpha):
    """Return the negative log-likelihood of a negative binomial.

    Vectorized version.

    Parameters
    ----------
    y : ndarray
        Observations.

    mu : float
        Mean of the distribution.

    alpha : float
        Dispersion of the distribution, s.t. the variance is
        :math:`\\mu + \\alpha * \\mu^2`.

    Returns
    -------
    float
        Negative log likelihood of the observations y following
        :math:`NB(\\mu, \\alpha)`.
    """

    n = len(y)
    alpha_neg1 = 1 / alpha
    logbinom = (
        gammaln(y[:, None] + alpha_neg1) - gammaln(y + 1)[:, None] - gammaln(alpha_neg1)
    )

    if len(mu.shape) == 1:
        return n * alpha_neg1 * np.log(alpha) + (
            -logbinom
            + (y[:, None] + alpha_neg1) * np.log(mu[:, None] + alpha_neg1)
            - (y * np.log(mu))[:, None]
        ).sum(0)
    else:
        return n * alpha_neg1 * np.log(alpha) + (
            -logbinom
            + (y[:, None] + alpha_neg1) * np.log(mu + alpha_neg1)
            - (y[:, None] * np.log(mu))
        ).sum(0)


def grid_fit_alpha(
    y,
    X,
    mu,
    alpha_hat,
    min_disp,
    max_disp,
    prior_disp_var=None,
    cr_reg=True,
    prior_reg=False,
    grid_length=100,
):
    """Find best dispersion parameter.

    Performs 1D grid search to maximize negative binomial log-likelihood.

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

    grid_length : int
        Number of grid points. (default: 100).

    Returns
    -------
    float
        Logarithm of the fitted dispersion parameter.
    """

    min_log_alpha = np.log(min_disp)
    max_log_alpha = np.log(max_disp)
    grid = np.linspace(min_log_alpha, max_log_alpha, grid_length)

    def loss(log_alpha):
        # closure to minimize
        alpha = np.exp(log_alpha)
        W = mu[:, None] / (1 + mu[:, None] * alpha)
        reg = 0
        if cr_reg:
            reg += (
                0.5 * np.linalg.slogdet((X.T[:, :, None] * W).transpose(2, 0, 1) @ X)[1]
            )
        if prior_reg:
            reg += (np.log(alpha) - np.log(alpha_hat)) ** 2 / (2 * prior_disp_var)
        return vec_nb_nll(y, mu, alpha) + reg

    ll_grid = loss(grid)

    min_idx = np.argmin(ll_grid)
    delta = grid[1] - grid[0]
    fine_grid = np.linspace(grid[min_idx] - delta, grid[min_idx] + delta, grid_length)

    ll_grid = loss(fine_grid)

    min_idx = np.argmin(ll_grid)
    log_alpha = fine_grid[min_idx]
    return log_alpha


def grid_fit_beta(
    counts,
    size_factors,
    X,
    disp,
    min_mu=0.5,
    grid_length=60,
    min_beta=-30,
    max_beta=30,
):
    """Find best LFC parameter.

    Perform 2D grid search to maximize negative binomial
    GLM log-likelihood w.r.t. LFCs.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    size_factors : pandas.Series
        DESeq2 normalization factors.

    X : ndarray
        Design matrix.

    disp : float
        Gene-wise dispersion prior.

    min_mu : float
        Lower threshold for dispersion parameters.

    grid_length : int
        Number of grid points. (default: 100).

    min_beta : int
        Lower-bound on LFC. (default: 30).

    max_beta : int
        Upper-bound on LFC. (default: 30).

    Returns
    -------
    float
        Fitted LFC parameter.
    """
    x_grid = np.linspace(min_beta, max_beta, grid_length)
    y_grid = np.linspace(min_beta, max_beta, grid_length)
    ll_grid = np.zeros((grid_length, grid_length))

    def loss(beta):
        # closure to minimize
        mu = np.maximum(size_factors[:, None] * np.exp(X @ beta.T), min_mu)
        return vec_nb_nll(counts, mu, disp) + 0.5 * (1e-6 * beta**2).sum(1)

    for i, x in enumerate(x_grid):
        ll_grid[i, :] = loss(np.array([[x, y] for y in y_grid]))

    min_idxs = np.unravel_index(np.argmin(ll_grid, axis=None), ll_grid.shape)
    delta = x_grid[1] - x_grid[0]

    fine_x_grid = np.linspace(
        x_grid[min_idxs[0]] - delta, x_grid[min_idxs[0]] + delta, grid_length
    )

    fine_y_grid = np.linspace(
        y_grid[min_idxs[1]] - delta,
        y_grid[min_idxs[1]] + delta,
        grid_length,
    )

    for i, x in enumerate(fine_x_grid):
        ll_grid[i, :] = loss(np.array([[x, y] for y in fine_y_grid]))

    min_idxs = np.unravel_index(np.argmin(ll_grid, axis=None), ll_grid.shape)
    beta = np.array([fine_x_grid[min_idxs[0]], fine_y_grid[min_idxs[1]]])
    return beta


def grid_fit_shrink_beta(
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
):
    """Find best LFC parameter.

    Performs 2D grid search to maximize MAP negative binomial
    GLM log-likelihood w.r.t. LFCs, with apeGLM prior.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    offset : ndarray
        Natural logarithm of size factor.

    design_matrix : ndarray
        Design matrix.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the LFC coefficient.

    scale_cnst : float
        Scaling factor for the optimization.

    grid_length : int
        Number of grid points. (default: 100).

    min_beta : int
        Lower-bound on LFC. (default: 30).

    max_beta : int
        Upper-bound on LFC. (default: 30).

    Returns
    -------
    float
        Fitted MAP LFC parameter.
    """
    x_grid = np.linspace(min_beta, max_beta, grid_length)
    y_grid = np.linspace(min_beta, max_beta, grid_length)
    ll_grid = np.zeros((grid_length, grid_length))

    def loss(beta):
        # closure to minimize
        return (
            pydeseq2.utils.nbinomFn(
                beta,
                design_matrix,
                counts,
                size,
                offset,
                prior_no_shrink_scale,
                prior_scale,
            )
            / scale_cnst
        )

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            ll_grid[i, j] = loss(np.array([x, y]))

    min_idxs = np.unravel_index(np.argmin(ll_grid, axis=None), ll_grid.shape)
    delta = x_grid[1] - x_grid[0]

    fine_x_grid = np.linspace(
        x_grid[min_idxs[0]] - delta, x_grid[min_idxs[0]] + delta, grid_length
    )

    fine_y_grid = np.linspace(
        y_grid[min_idxs[1]] - delta,
        y_grid[min_idxs[1]] + delta,
        grid_length,
    )

    for i, x in enumerate(fine_x_grid):
        for j, y in enumerate(fine_y_grid):
            ll_grid[i, j] = loss(np.array([x, y]))

    min_idxs = np.unravel_index(np.argmin(ll_grid, axis=None), ll_grid.shape)
    beta = np.array([fine_x_grid[min_idxs[0]], fine_y_grid[min_idxs[1]]])
    return beta
