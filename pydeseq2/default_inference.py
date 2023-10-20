import warnings
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from joblib import Parallel  # type: ignore
from joblib import delayed
from joblib import parallel_backend
from statsmodels.tools.sm_exceptions import DomainWarning  # type: ignore

from pydeseq2 import inference
from pydeseq2 import utils

# Ignore DomainWarning raised by statsmodels when fitting a Gamma GLM with identity link.
warnings.simplefilter("ignore", DomainWarning)


class DefaultInference(inference.Inference):
    """Default DESeq2-related inference methods, using scipy/sklearn/numpy."""

    fit_rough_dispersions = staticmethod(utils.fit_rough_dispersions)
    fit_moments_dispersions = staticmethod(utils.fit_moments_dispersions)

    def __init__(
        self,
        joblib_verbosity: int = 0,
        batch_size: int = 128,
        n_cpus: Optional[int] = None,
        backend: str = "loky",
    ):
        self.joblib_verbosity = joblib_verbosity
        self.batch_size = batch_size
        self.n_processes = utils.get_num_processes(n_cpus)
        self.backend = backend

    def lin_reg_mu(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        with parallel_backend(self.backend, inner_max_num_threads=1):
            mu_hat_ = np.array(
                Parallel(
                    n_jobs=self.n_processes,
                    verbose=self.joblib_verbosity,
                    batch_size=self.batch_size,
                )(
                    delayed(utils.fit_lin_mu)(
                        counts=counts[:, i],
                        size_factors=size_factors,
                        design_matrix=design_matrix,
                        min_mu=min_mu,
                    )
                    for i in range(counts.shape[1])
                )
            )
        return mu_hat_.T

    def irls(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        min_mu: float,
        beta_tol: float,
    ):
        with parallel_backend(self.backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(utils.irls_solver)(
                    counts=counts[:, i],
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    disp=disp[i],
                    min_mu=min_mu,
                    beta_tol=beta_tol,
                )
                for i in range(counts.shape[1])
            )
        res = zip(*res)
        MLE_lfcs_, mu_hat_, hat_diagonals_, converged_ = (np.array(m) for m in res)

        return (
            MLE_lfcs_,
            mu_hat_.T,
            hat_diagonals_.T,
            converged_,
        )

    def alpha_mle(
        self,
        counts: np.ndarray,
        design_matrix: np.ndarray,
        mu: np.ndarray,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        prior_disp_var: Optional[float] = None,
        cr_reg: bool = True,
        prior_reg: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with parallel_backend(self.backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(utils.fit_alpha_mle)(
                    counts=counts[:, i],
                    design_matrix=design_matrix,
                    mu=mu[:, i],
                    alpha_hat=alpha_hat[i],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    prior_disp_var=prior_disp_var,
                    cr_reg=cr_reg,
                    prior_reg=prior_reg,
                )
                for i in range(counts.shape[1])
            )
        res = zip(*res)
        dispersions_, l_bfgs_b_converged_ = (np.array(m) for m in res)
        return dispersions_, l_bfgs_b_converged_

    def wald_test(
        self,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        lfc: np.ndarray,
        mu: np.ndarray,
        ridge_factor: np.ndarray,
        contrast: np.ndarray,
        lfc_null: np.ndarray,
        alt_hypothesis: Optional[
            Literal["greaterAbs", "lessAbs", "greater", "less"]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_genes = mu.shape[1]
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(utils.wald_test)(
                    design_matrix=design_matrix,
                    disp=disp[i],
                    lfc=lfc[i],
                    mu=mu[:, i],
                    ridge_factor=ridge_factor,
                    contrast=contrast,
                    lfc_null=lfc_null,  # Convert log2 to natural log
                    alt_hypothesis=alt_hypothesis,
                )
                for i in range(num_genes)
            )
        res = zip(*res)
        pvals, stats, se = (np.array(m) for m in res)

        return pvals, stats, se

    def dispersion_trend_gamma_glm(
        self, covariates: pd.Series, targets: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        covariates_w_intercept = sm.add_constant(covariates)
        targets_fit = targets.values
        covariates_fit = covariates_w_intercept.values
        glm_gamma = sm.GLM(
            targets_fit,
            covariates_fit,
            family=sm.families.Gamma(link=sm.families.links.identity()),
        )
        res = glm_gamma.fit()
        coeffs = res.params
        return (coeffs, covariates_fit @ coeffs)

    def lfc_shrink_nbinom_glm(
        self,
        design_matrix: np.ndarray,
        counts: np.ndarray,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with parallel_backend(self.backend, inner_max_num_threads=1):
            num_genes = counts.shape[1]
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(utils.nbinomGLM)(
                    design_matrix=design_matrix,
                    counts=counts[:, i],
                    size=size[i],
                    offset=offset,
                    prior_no_shrink_scale=prior_no_shrink_scale,
                    prior_scale=prior_scale,
                    optimizer=optimizer,
                    shrink_index=shrink_index,
                )
                for i in range(num_genes)
            )
        res = zip(*res)
        lfcs, inv_hessians, l_bfgs_b_converged_ = (np.array(m) for m in res)
        return lfcs, inv_hessians, l_bfgs_b_converged_