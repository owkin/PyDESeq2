import anndata as ad  # type: ignore
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
import numpy as np
import statsmodels.api as sm  # type: ignore
from jax import jit
from jax import vmap
from jaxopt import ScipyBoundedMinimize
from joblib import Parallel  # type: ignore
from joblib import delayed
from joblib import parallel_backend
from scipy.linalg import solve  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.special import polygamma  # type: ignore
from scipy.stats import f  # type: ignore
from scipy.stats import norm  # type: ignore
from scipy.stats import trim_mean  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from statsmodels.tools.sm_exceptions import DomainWarning  # type: ignore


def make_gram_matrix_cho_factor(design_matrix: np.ndarray) -> np.ndarray:
    """Compute the Cholesky factor of the Gram matrix of a design matrix.

    Parameters
    ----------
    design_matrix : ndarray
        Design matrix.

    Returns
    -------
    ndarray
        Cholesky factor of the Gram matrix.
    """
    gram_matrix = jnp.matmul(design_matrix.T, design_matrix)
    return jsp.linalg.cho_factor(gram_matrix)[0]


def fit_lin_mu_jax(
    gram_matrix_cho_factor: np.ndarray,
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    size_factors : ndarray
        Sample-wise scaling factors (obtained from median-of-ratios).

    design_matrix : ndarray
        Design matrix.

    min_mu : float
        Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

    Returns
    -------
    ndarray
        Estimated mean.
    """
    new_counts = jnp.reshape(counts / size_factors, (counts.shape[0], 1))
    z = jnp.matmul(jnp.transpose(design_matrix), new_counts)
    beta = jsp.linalg.cho_solve(
        (gram_matrix_cho_factor, False), z, overwrite_b=False, check_finite=True
    )

    mu_hat = jnp.multiply(size_factors, jnp.ravel(jnp.matmul(design_matrix, beta)))
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return jnp.maximum(mu_hat, min_mu)


def fit_lin_mu_for_all_genes_jax(
    full_counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
):
    gram_matrix_cho_factor = make_gram_matrix_cho_factor(design_matrix)
    mu_hat = vmap(fit_lin_mu_jax, in_axes=(None, 1, None, None, None), out_axes=1)(
        gram_matrix_cho_factor,
        full_counts,
        size_factors,
        design_matrix,
        min_mu,
    )
    return mu_hat


def fit_alpha_mle_single_gene_jax(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    n_samples: int,
    alpha_hat: float,
    min_disp: float,
    max_disp: float,
    cr_reg: bool = True,
) -> tuple[float, bool]:
    """Estimate the dispersion parameter of a negative binomial GLM.

    Note: it is possible to pass counts, design_matrix and mu arguments in the form of
    pandas Series, but using numpy arrays makes the code significantly faster.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    design_matrix : ndarray
        Design matrix.

    mu : ndarray
        Mean estimation for the NB model.

    alpha_hat : float
        Initial dispersion estimate.

    min_disp : float
        Lower threshold for dispersion parameters.

    max_disp : float
        Upper threshold for dispersion parameters.


    cr_reg : bool
        Whether to use Cox-Reid regularization. (default: ``True``).

    optimizer : str
        Optimizing method to use. Accepted values: 'BFGS' or 'L-BFGS-B'.
        (default: ``'L-BFGS-B'``).

    Returns
    -------
    float
        Dispersion estimate.

    bool
        Whether L-BFGS-B converged. If not, dispersion is estimated using grid search.
    """

    log_alpha_hat = jnp.log(alpha_hat)

    def loss_without_grad(log_alpha: float, *args, **kwargs) -> float:
        # closure to be minimized
        alpha = jnp.exp(log_alpha)
        if cr_reg:
            W = jnp.divide(mu, (1 + jnp.multiply(mu, alpha)))
            reg = (
                0.5
                * jnp.linalg.slogdet(
                    jnp.matmul(
                        jnp.multiply(jnp.transpose(design_matrix), W), design_matrix
                    )
                )[1]
            )

        else:
            reg = 0
        return nb_nll_jax(counts, mu, alpha, n=n_samples) + reg

    def loss_with_grad(log_alpha: float, *args, **kwargs) -> tuple[float, float]:
        # closure to be minimized
        alpha = jnp.exp(log_alpha)
        if cr_reg:
            W = jnp.divide(mu, (1 + jnp.multiply(mu, alpha)))
            reg = (
                0.5
                * jnp.linalg.slogdet(
                    jnp.matmul(
                        jnp.multiply(jnp.transpose(design_matrix), W), design_matrix
                    )
                )[1]
            )
            dW = -jnp.power(W, 2)
            reg_grad = jnp.multiply(
                0.5 * alpha,
                jnp.sum(
                    jnp.multiply(
                        jnp.linalg.inv(
                            jnp.matmul(
                                jnp.multiply(jnp.transpose(design_matrix), W),
                                design_matrix,
                            )
                        ),
                        jnp.matmul(
                            jnp.multiply(jnp.transpose(design_matrix), dW),
                            design_matrix,
                        ),
                    )
                ),
            )  # since we want the gradient wrt log_alpha,
            # we need to multiply by alpha

        else:
            reg = 0
            reg_grad = 0
        return (
            nb_nll_jax(counts, mu, alpha, n=n_samples) + reg,
            alpha * dnb_nll_jax(counts, mu, alpha) + reg_grad,
        )

    lbfgsb = jaxopt.LBFGSB(fun=loss_without_grad, value_and_grad=False, has_aux=False)
    param = lbfgsb.run(
        log_alpha_hat,
        bounds=(
            jnp.log(min_disp),
            jnp.log(max_disp),
        ),
    ).params

    return param


def alpha_ll_with_grad_jax(
    log_alpha: float,
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    n_samples: int,
):
    # closure to be minimized
    alpha = jnp.exp(log_alpha)
    W = jnp.divide(mu, (1 + jnp.multiply(mu, alpha)))
    reg = (
        0.5
        * jnp.linalg.slogdet(
            jnp.matmul(jnp.multiply(jnp.transpose(design_matrix), W), design_matrix)
        )[1]
    )
    dW = -jnp.power(W, 2)
    reg_grad = jnp.multiply(
        0.5 * alpha,
        jnp.sum(
            jnp.multiply(
                jnp.linalg.inv(
                    jnp.matmul(
                        jnp.multiply(jnp.transpose(design_matrix), W),
                        design_matrix,
                    )
                ),
                jnp.matmul(
                    jnp.multiply(jnp.transpose(design_matrix), dW),
                    design_matrix,
                ),
            )
        ),
    )  # since we want the gradient wrt log_alpha,
    # we need to multiply by alpha

    return (
        nb_nll_jax(counts, mu, alpha, n=n_samples) + reg,
        alpha * dnb_nll_jax(counts, mu, alpha) + reg_grad,
    )


def alpha_ll_single_gene_jax(
    log_alpha: float,
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    n_samples: int,
) -> tuple[float, bool]:
    """Estimate the dispersion parameter of a negative binomial GLM.

    Note: it is possible to pass counts, design_matrix and mu arguments in the form of
    pandas Series, but using numpy arrays makes the code significantly faster.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    design_matrix : ndarray
        Design matrix.

    mu : ndarray
        Mean estimation for the NB model.


    Returns
    -------
    float
        Dispersion estimate.
    """
    # closure to be minimized
    alpha = jnp.exp(log_alpha)
    W = jnp.divide(mu, (1 + jnp.multiply(mu, alpha)))
    reg = (
        0.5
        * jnp.linalg.slogdet(
            jnp.matmul(jnp.multiply(jnp.transpose(design_matrix), W), design_matrix)
        )[1]
    )

    return nb_nll_jax(counts, mu, alpha, n=n_samples) + reg


def fit_alpha_mle_single_gene_jax_v2(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    n_samples: int,
    alpha_hat: float,
    min_disp: float,
    max_disp: float,
) -> tuple[float, bool]:
    """Estimate the dispersion parameter of a negative binomial GLM.

    Note: it is possible to pass counts, design_matrix and mu arguments in the form of
    pandas Series, but using numpy arrays makes the code significantly faster.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    design_matrix : ndarray
        Design matrix.

    mu : ndarray
        Mean estimation for the NB model.

    alpha_hat : float
        Initial dispersion estimate.

    min_disp : float
        Lower threshold for dispersion parameters.

    max_disp : float
        Upper threshold for dispersion parameters.


    cr_reg : bool
        Whether to use Cox-Reid regularization. (default: ``True``).

    optimizer : str
        Optimizing method to use. Accepted values: 'BFGS' or 'L-BFGS-B'.
        (default: ``'L-BFGS-B'``).

    Returns
    -------
    float
        Dispersion estimate.

    bool
        Whether L-BFGS-B converged. If not, dispersion is estimated using grid search.
    """

    log_alpha_hat = jnp.log(alpha_hat)

    lbfgsb = jaxopt.LBFGSB(
        fun=alpha_ll_with_grad_jax,
        value_and_grad=True,
        verbose=False,
        maxiter=50,
        stop_if_linesearch_fails=True,
    )
    param, state = lbfgsb.run(
        log_alpha_hat,
        counts=counts,
        design_matrix=design_matrix,
        mu=mu,
        n_samples=n_samples,
        bounds=(jnp.log(min_disp), jnp.log(max_disp)),
    )

    return jnp.exp(param), state.grad, state.failed_linesearch


def fit_alpha_mle_all_genes_jax_v2(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    n_samples: int,
    alpha_hat: np.ndarray,
    min_disp: float,
    max_disp: float,
):

    param, state_grad, state_ls = vmap(
        fit_alpha_mle_single_gene_jax_v2,
        in_axes=(1, None, 1, None, 0, None, None),
        out_axes=(0, 0, 0),
    )(counts, design_matrix, mu, n_samples, alpha_hat, min_disp, max_disp)
    return param, state_grad, state_ls


def fit_alpha_mle_all_genes_jax(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    n_samples: int,
    alpha_hat: np.ndarray,
    min_disp: float,
    max_disp: float,
    cr_reg: bool = True,
):

    param = vmap(
        fit_alpha_mle_single_gene_jax,
        in_axes=(1, None, 1, None, 0, None, None, None),
        out_axes=0,
    )(counts, design_matrix, mu, n_samples, alpha_hat, min_disp, max_disp, cr_reg)
    return param


def dnb_nll_jax(counts: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Gradient of the negative log-likelihood of a negative binomial.

    Unvectorized.

    Parameters
    ----------
    counts : ndarray
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

    alpha_neg1 = jnp.divide(1, alpha)
    ll_part = jnp.power(alpha_neg1, 2) * jnp.sum(
        jsp.special.polygamma(0, alpha_neg1)
        - jsp.special.polygamma(0, counts + alpha_neg1)
        + jnp.log(1 + mu * alpha)
        + jnp.divide((counts - mu), (mu + alpha_neg1))
    )

    return -ll_part


def nb_nll_jax(
    counts: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    n: int,
) -> np.ndarray:
    """We assume that alpha has the same shape as the counts."""
    alpha_neg1 = jnp.divide(1, alpha)
    logbinom = (
        jsp.special.gammaln(counts + alpha_neg1)
        - jsp.special.gammaln(counts + 1)
        - jsp.special.gammaln(alpha_neg1)
    )
    result = n * alpha_neg1 * jnp.log(alpha) + jnp.sum(
        -logbinom
        + (counts + alpha_neg1) * jnp.log(alpha_neg1 + mu)
        - counts * jnp.log(mu)
    )
    return result
