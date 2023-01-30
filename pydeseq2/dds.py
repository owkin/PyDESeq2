import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from scipy.special import polygamma
from scipy.stats import f
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import DomainWarning

from pydeseq2.preprocessing import deseq2_norm
from pydeseq2.utils import build_design_matrix
from pydeseq2.utils import dispersion_trend
from pydeseq2.utils import fit_alpha_mle
from pydeseq2.utils import fit_lin_mu
from pydeseq2.utils import fit_moments_dispersions
from pydeseq2.utils import fit_rough_dispersions
from pydeseq2.utils import get_num_processes
from pydeseq2.utils import irls_solver
from pydeseq2.utils import robust_method_of_moments_disp
from pydeseq2.utils import test_valid_counts
from pydeseq2.utils import trimmed_mean

# Ignore DomainWarning raised by statsmodels when fitting a Gamma GLM with identity link.
warnings.simplefilter("ignore", DomainWarning)
# Ignore AnnData's FutureWarning about implicit data conversion.
warnings.simplefilter("ignore", FutureWarning)

# TODO: support loading / saving DeseqDataSets from AnnData objects.


class DeseqDataSet(ad.AnnData):
    r"""A class to implement dispersion and log fold-change (LFC) estimation.

    The DeseqDataSet extends the `AnnData class
    <https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html#anndata.AnnData>`_.
    As such, it implements the same methods and attributes, in addition to those that are
    specific to pydeseq2.
    Dispersions and LFCs are estimated following the DESeq2 pipeline
    :cite:p:`DeseqDataSet-love2014moderated`.

    Parameters
    ----------
    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.

    clinical : pandas.DataFrame
        DataFrame containing clinical information.
        Must be indexed by sample barcodes.

    design_factors : str or list[str]
        Name of the columns of clinical to be used as design variables. If a list,
        the last factor will be considered the variable of interest by default.
        Only bi-level factors are supported. (default: 'condition').

    reference_level : str
        The factor to use as a reference. Must be one of the values taken by the design.
        If None, the reference will be chosen alphabetically (last in order).
        (default: None).

    min_mu : float
        Threshold for mean estimates. (default: 0.5).

    min_disp : float
        Lower threshold for dispersion parameters. (default: 1e-8).

    max_disp : float
        Upper threshold for dispersion parameters.
        NB: The threshold that is actually enforced is max(max_disp, len(counts)).
        (default: 10).

    refit_cooks : bool
        Whether to refit cooks outliers. (default: True).

    min_replicates : int
        Minimum number of replicates a condition should have
        to allow refitting its samples. (default: 7).

    beta_tol : float
        Stopping criterion for IRWLS:
        math:: abs(dev - old_dev) / (abs(dev) + 0.1) < beta_tol.
        (default: 1e-8).

    n_cpus : int
        Number of cpus to use. If None, all available cpus will be used. (default: None).

    batch_size : int
        Number of tasks to allocate to each joblib parallel worker. (default: 128).

    joblib_verbosity : int
        The verbosity level for joblib tasks. The higher the value, the more updates
        are reported. (default: 0).

    Attributes
    ----------
    X
        A ‘number of samples’ x ‘number of genes’ count data matrix.

    obs
        Key-indexed one-dimensional observations annotation of length 'number of
        samples". Used to store design factors.

    var
        Key-indexed one-dimensional gene-level annotation of length ‘number of genes’.

    uns
        Key-indexed unstructured annotation.

    obsm
        Key-indexed multi-dimensional observations annotation of length
        ‘number of samples’. Stores "design_matrix" and "size_factors", among others.

    varm
        Key-indexed multi-dimensional gene annotation of length #variables.
        Stores "dispersions" and "LFC", among others.

    layers
        Key-indexed multi-dimensional arrays aligned to dimensions of `X`, e.g. "cooks".

    n_processes : int
        Number of cpus to use for multiprocessing.

    non_zero : ndarray
        Boolean mask of genes that have non-uniformly zero counts.

    non_zero_genes : pandas.Index
        Index of genes that have non-uniformly zero counts.

    counts_to_refit : anndata.AnnData
        Read counts after replacement, containing only genes
        for which dispersions and LFCs must be fitted again.

    new_all_zeroes_genes : pandas.Index
        Genes which have only zero counts after outlier replacement.

    References
    ----------
    .. bibliography::
        :keyprefix: DeseqDataSet-
    """

    def __init__(
        self,
        counts,
        clinical,
        design_factors="condition",
        reference_level=None,
        min_mu=0.5,
        min_disp=1e-8,
        max_disp=10.0,
        refit_cooks=True,
        min_replicates=7,
        beta_tol=1e-8,
        n_cpus=None,
        batch_size=128,
        joblib_verbosity=0,
    ):

        # Test counts before going further
        test_valid_counts(counts)

        # Initialize the AnnData part
        super().__init__(X=counts, obs=clinical, dtype=int)

        # Convert design_factors to list if a single string was provided.
        self.design_factors = (
            [design_factors] if isinstance(design_factors, str) else design_factors
        )
        if self.obs[self.design_factors].isna().any().any():
            raise ValueError("NaNs are not allowed in the design factors.")
        self.obs[self.design_factors] = self.obs[self.design_factors].astype(str)

        # Build the design matrix
        # Stored in the obsm attribute of the dataset
        self.obsm["design_matrix"] = build_design_matrix(
            clinical_df=self.obs,
            design_factors=self.design_factors,
            ref=reference_level,
            expanded=False,
            intercept=True,
        )

        self.min_mu = min_mu
        self.min_disp = min_disp
        self.max_disp = np.maximum(max_disp, self.n_obs)
        self.refit_cooks = refit_cooks
        self.min_replicates = min_replicates
        self.beta_tol = beta_tol
        self.n_processes = get_num_processes(n_cpus)
        self.batch_size = batch_size
        self.joblib_verbosity = joblib_verbosity

    def deseq2(self):
        """Perform dispersion and log fold-change (LFC) estimation.

        Wrapper for the first part of the PyDESeq2 pipeline.
        """

        # Compute DESeq2 normalization factors using the Median-of-ratios method
        self.fit_size_factors()
        # Fit an independent negative binomial model per gene
        self.fit_genewise_dispersions()
        # Fit a parameterized trend curve for dispersions, of the form
        # math:: f(\mu) = \alpha_1/\mu + a_0
        self.fit_dispersion_trend()
        # Compute prior dispersion variance
        self.fit_dispersion_prior()
        # Refit genewise dispersions a posteriori (shrinks estimates towards trend curve)
        self.fit_MAP_dispersions()
        # Fit log-fold changes (in natural log scale)
        self.fit_LFC()
        # Compute Cooks distances to find outliers
        self.calculate_cooks()

        if self.refit_cooks:
            # Replace outlier counts, and refit dispersions and LFCs
            # for genes that had outliers replaced
            self.refit()

    def fit_size_factors(self):
        """Fit sample-wise deseq2 normalization (size) factors.

        Uses the median-of-ratios method.
        """
        print("Fitting size factors...")
        start = time.time()
        _, self.obsm["size_factors"] = deseq2_norm(self.X)
        end = time.time()
        print(f"... done in {end - start:.2f} seconds.\n")

    def fit_genewise_dispersions(self):
        """Fit gene-wise dispersion estimates.

        Fits a negative binomial per gene, independently.
        """
        # Check that size factors are available. If not, compute them.
        if "size_factors" not in self.obsm:
            self.fit_size_factors()

        # Fit "method of moments" dispersion estimates
        self._fit_MoM_dispersions()

        # Exclude genes with all zeroes
        # TODO : this could probably be simplified
        self.non_zero = ~(self.X == 0).all(
            axis=0
        )  # TODO: add docstring for this attribute
        self.non_zero_genes = self.var_names[self.non_zero]
        nz_data = self[:, self.non_zero_genes]
        num_genes = nz_data.n_vars

        # Convert design_matrix to numpy for speed
        X = nz_data.obsm["design_matrix"].values
        rough_disps = nz_data.varm["_rough_dispersions"]
        size_factors = nz_data.obsm["size_factors"]

        # mu_hat is initialized differently depending on the number of different factor
        # groups. If there are as many different factor combinations as design factors
        # (intercept included), it is fitted with a linear model, otherwise it is fitted
        # with a GLM (using rough dispersion estimates).
        if len(nz_data.obsm["design_matrix"].value_counts()) == self.n_vars:
            with parallel_backend("loky", inner_max_num_threads=1):
                mu_hat_ = np.array(
                    Parallel(
                        n_jobs=self.n_processes,
                        verbose=self.joblib_verbosity,
                        batch_size=self.batch_size,
                    )(
                        delayed(fit_lin_mu)(
                            counts=nz_data.X[:, i],
                            size_factors=size_factors,
                            design_matrix=X,
                            min_mu=self.min_mu,
                        )
                        for i in range(num_genes)
                    )
                )
        else:
            with parallel_backend("loky", inner_max_num_threads=1):
                res = Parallel(
                    n_jobs=self.n_processes,
                    verbose=self.joblib_verbosity,
                    batch_size=self.batch_size,
                )(
                    delayed(irls_solver)(
                        counts=nz_data.X[:, i],
                        size_factors=size_factors,
                        design_matrix=X,
                        disp=rough_disps[i],
                        min_mu=self.min_mu,
                        beta_tol=self.beta_tol,
                    )
                    for i in range(num_genes)
                )

                _, mu_hat_, _, _ = zip(*res)
                mu_hat_ = np.array(mu_hat_)

        # TODO : do this better

        _mu_hat = pd.DataFrame(np.NaN, index=self.obs_names, columns=self.var_names)

        _mu_hat.update(
            pd.DataFrame(mu_hat_.T, index=self.obs_names, columns=self.non_zero_genes)
        )

        self.layers["_mu_hat"] = _mu_hat.values

        print("Fitting dispersions...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(fit_alpha_mle)(
                    counts=nz_data.X[:, i],
                    design_matrix=X,
                    mu=mu_hat_[i, :],
                    alpha_hat=rough_disps[i],
                    min_disp=self.min_disp,
                    max_disp=self.max_disp,
                )
                for i in range(num_genes)
            )
        end = time.time()
        print(f"... done in {end - start:.2f} seconds.\n")

        dispersions_, l_bfgs_b_converged_ = zip(*res)

        # TODO : this might not be the best way to add data to an anndata...
        # TODO : use np.where and such to do data allocations?
        # genewise_dispersions = pd.Series(np.NaN, index=self.var_names)
        # genewise_dispersions.update(
        #     pd.Series(dispersions_, index=self.non_zero_genes).clip(
        #         self.min_disp, self.max_disp
        #     )
        # )
        #
        # self.varm["genewise_dispersions"] = genewise_dispersions.values

        self.varm["genewise_dispersions"] = np.full(self.n_vars, np.NaN)
        self.varm["genewise_dispersions"][self.non_zero] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        # _genewise_converged = pd.Series(np.NaN, index=self.var_names)
        #
        # _genewise_converged.update(
        #     pd.Series(l_bfgs_b_converged_, index=self.non_zero_genes)
        # )
        #
        # self.varm["_genewise_converged"] = _genewise_converged.values

        self.varm["_genewise_converged"] = np.full(self.n_vars, np.NaN)
        self.varm["_genewise_converged"][self.non_zero] = np.array(l_bfgs_b_converged_)

    def fit_dispersion_trend(self):
        r"""Fit the dispersion trend coefficients.

        .. math:: f(\mu) = \alpha_1/\mu + a_0.
        """

        # Check that genewise dispersions are available. If not, compute them.
        # if not hasattr(self, "genewise_dispersions"):
        if not "genewise_dispersions" not in self.varm:
            self.fit_genewise_dispersions()

        print("Fitting dispersion trend curve...")
        start = time.time()
        self.varm["_normed_means"] = (self.X / self.obsm["size_factors"][:, None]).mean(
            0
        )

        # TODO : not a very good way to use pandas and annData (array to pandas)
        # Exclude all-zero counts
        # targets = self.genewise_dispersions.loc[self.non_zero_genes].copy()
        # covariates = sm.add_constant(1 / self._normed_means.loc[self.non_zero_genes])

        targets = pd.Series(
            self[:, self.non_zero_genes].varm["genewise_dispersions"].copy(),
            index=self.non_zero_genes,
        )
        covariates = sm.add_constant(
            pd.Series(
                1 / self[:, self.non_zero_genes].varm["_normed_means"],
                index=self.non_zero_genes,
            )
        )

        for gene in self.non_zero_genes:
            if (
                np.isinf(covariates.loc[gene]).any()
                or np.isnan(covariates.loc[gene]).any()
            ):
                targets.drop(labels=[gene], inplace=True)
                covariates.drop(labels=[gene], inplace=True)

        # Initialize coefficients
        old_coeffs = pd.Series([0.1, 0.1])
        coeffs = pd.Series([1.0, 1.0])

        while (np.log(np.abs(coeffs / old_coeffs)) ** 2).sum() >= 1e-6:

            glm_gamma = sm.GLM(
                targets.values,
                covariates.values,
                family=sm.families.Gamma(link=sm.families.links.identity()),
            )

            res = glm_gamma.fit()
            old_coeffs = coeffs.copy()
            coeffs = res.params

            # Filter out genes that are too far away from the curve before refitting
            predictions = covariates.values @ coeffs
            # pred_ratios = self.genewise_dispersions.loc[covariates.index] / predictions
            pred_ratios = (
                self[:, covariates.index].varm["genewise_dispersions"] / predictions
            )

            targets.drop(
                targets[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )
            covariates.drop(
                covariates[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )

        end = time.time()
        print(f"... done in {end - start:.2f} seconds.\n")

        # self.trend_coeffs = pd.Series(coeffs, index=["a0", "a1"])
        # self.fitted_dispersions = pd.Series(
        #     np.NaN, index=self.genewise_dispersions.index
        # )
        # self.fitted_dispersions.update(
        #     dispersion_trend(
        #         self._normed_means.loc[self.non_zero_genes], self.trend_coeffs
        #     )
        # )

        self.uns["trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])

        # TODO : again, maybe not the best way to add data
        # fitted_dispersions = pd.Series(np.NaN, index=self.var_names)
        # fitted_dispersions.update(
        #     pd.Series(
        #         dispersion_trend(
        #             self[:, self.non_zero_genes].varm["_normed_means"],
        #             self.uns["trend_coeffs"],
        #         ),
        #         index=self.non_zero_genes,
        #     )
        # )
        #
        # self.varm["fitted_dispersions"] = fitted_dispersions.values

        self.varm["fitted_dispersions"] = np.full(self.n_vars, np.NaN)
        self.varm["fitted_dispersions"][self.non_zero] = dispersion_trend(
            self.varm["_normed_means"][self.non_zero],
            self.uns["trend_coeffs"],
        )

    def fit_dispersion_prior(self):
        """Fit dispersion variance priors and standard deviation of log-residuals.

        The computation is based on genes whose dispersions are above 100 * min_disp.
        """

        # Check that the dispersion trend curve was fitted. If not, fit it.
        if "trend_coeffs" not in self.uns:
            self.fit_dispersion_trend()

        # Exclude genes with all zeroes
        num_genes = len(self.non_zero_genes)
        num_vars = self.n_vars

        # Fit dispersions to the curve, and compute log residuals
        disp_residuals = np.log(
            self[:, self.non_zero_genes].varm["genewise_dispersions"]
        ) - np.log(self[:, self.non_zero_genes].varm["fitted_dispersions"])

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = self[:, self.non_zero_genes].varm["genewise_dispersions"] >= (
            100 * self.min_disp
        )
        self.uns["_squared_logres"] = np.median(
            np.abs(disp_residuals[above_min_disp])
        ) ** 2 / norm.ppf(0.75)
        self.uns["prior_disp_var"] = np.maximum(
            self.uns["_squared_logres"] - polygamma(1, (num_genes - num_vars) / 2),
            0.25,
        )

    def fit_MAP_dispersions(self):
        """Fit Maximum a Posteriori dispersion estimates.

        After MAP dispersions are fit, filter genes for which we don't apply shrinkage.
        """

        # Check that the dispersion prior variance is available. If not, compute it.
        # if not hasattr(self, "prior_disp_var"):
        #     self.fit_dispersion_prior()

        if "prior_disp_var" not in self.uns:
            self.fit_dispersion_prior()

        # Exclude genes with all zeroes
        nz_data = self[:, self.non_zero_genes]
        num_genes = nz_data.n_vars

        X = nz_data.obsm["design_matrix"].values
        mu_hat = nz_data.layers["_mu_hat"]
        fit_disps = nz_data.varm["fitted_dispersions"]

        print("Fitting MAP dispersions...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(fit_alpha_mle)(
                    counts=nz_data.X[:, i],
                    design_matrix=X,
                    mu=mu_hat[:, i],
                    alpha_hat=fit_disps[i],
                    min_disp=self.min_disp,
                    max_disp=self.max_disp,
                    prior_disp_var=self.uns["prior_disp_var"].item(),
                    cr_reg=True,
                    prior_reg=True,
                )
                for i in range(num_genes)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        dispersions_, l_bfgs_b_converged_ = zip(*res)

        self.varm["MAP_dispersions"] = np.full(self.n_vars, np.NaN)
        self.varm["MAP_dispersions"][self.non_zero] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_MAP_converged"] = np.full(self.n_vars, np.NaN)
        self.varm["_MAP_converged"][self.non_zero] = l_bfgs_b_converged_

        # Filter outlier genes for which we won't apply shrinkage
        self.varm["dispersions"] = self.varm["MAP_dispersions"].copy()
        self.varm["_outlier_genes"] = np.log(self.varm["dispersions"]) > np.log(
            self.varm["fitted_dispersions"]
        ) + 2 * np.sqrt(self.uns["_squared_logres"])
        self.varm["dispersions"][self.varm["_outlier_genes"]] = self.varm[
            "genewise_dispersions"
        ][self.varm["_outlier_genes"]]

    def fit_LFC(self):
        """Fit log fold change (LFC) coefficients.

        In the 2-level setting, the intercept corresponds to the base mean,
        while the second is the actual LFC coefficient, in natural log scale.
        """

        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()

        # Exclude genes with all zeroes
        nz_data = self[:, self.non_zero_genes]
        num_genes = nz_data.n_vars

        X = nz_data.obsm["design_matrix"].values
        size_factors = nz_data.obsm["size_factors"]
        disps = nz_data.varm["dispersions"]

        print("Fitting LFCs...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(irls_solver)(
                    counts=nz_data.X[:, i],
                    size_factors=size_factors,
                    design_matrix=X,
                    disp=disps[i],
                    min_mu=self.min_mu,
                    beta_tol=self.beta_tol,
                )
                for i in range(num_genes)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        MLE_lfcs_, mu_, hat_diagonals_, converged_ = zip(*res)

        self.varm["LFC"] = pd.DataFrame(
            np.NaN,
            index=self.var_names,
            columns=self.obsm["design_matrix"].columns,
        )

        self.varm["LFC"].update(
            pd.DataFrame(
                MLE_lfcs_,
                index=self.non_zero_genes,
                columns=self.obsm["design_matrix"].columns,
            )
        )

        _mu_LFC = pd.DataFrame(np.NaN, index=self.obs_names, columns=self.var_names)
        _mu_LFC.update(
            pd.DataFrame(
                np.array(mu_).T, index=self.obs_names, columns=self.non_zero_genes
            )
        )

        self.layers["_mu_LFC"] = _mu_LFC.values

        _hat_diagonals = pd.DataFrame(
            np.NaN,
            index=self.obs_names,
            columns=self.var_names,
        )
        _hat_diagonals.update(
            pd.DataFrame(
                np.array(hat_diagonals_).T,
                index=self.obs_names,
                columns=self.non_zero_genes,
            )
        )

        self.layers["_hat_diagonals"] = _hat_diagonals.values

        _LFC_converged = pd.Series(np.NaN, index=self.var_names)
        _LFC_converged.update(pd.Series(converged_, index=self.non_zero_genes))

        self.varm["_LFC_converged"] = _LFC_converged.values

    def calculate_cooks(self):
        """Compute Cook's distance for outlier detection.

        Measures the contribution of a single entry to the output of LFC estimation.
        """

        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()

        num_vars = self.n_vars
        nz_data = self[:, self.non_zero_genes]

        # Keep only non-zero genes
        # TODO: used a DataFrame to avoid changing robust_method_of_moments_disp
        # but we should be able to pass an array
        normed_counts = pd.DataFrame(
            nz_data.X / self.obsm["size_factors"][:, None],
            index=self.obs_names,
            columns=self.non_zero_genes,
        )

        # dispersions = pd.Series(np.NaN, index=self.var_names)
        dispersions = robust_method_of_moments_disp(
            normed_counts, self.obsm["design_matrix"]
        )

        V = (
            nz_data.layers["_mu_LFC"]
            + dispersions.values[None, :] * nz_data.layers["_mu_LFC"] ** 2
        )
        squared_pearson_res = (nz_data.X - nz_data.layers["_mu_LFC"]) ** 2 / V

        cooks = pd.DataFrame(np.NaN, index=self.obs_names, columns=self.var_names)
        cooks.update(
            pd.DataFrame(
                squared_pearson_res
                / num_vars
                * (
                    nz_data.layers["_hat_diagonals"]
                    / (1 - nz_data.layers["_hat_diagonals"]) ** 2
                ),
                index=self.obs_names,
                columns=self.non_zero_genes,
            )
        )

        self.layers["cooks"] = cooks.values

    def refit(self):
        """Refit Cook outliers.

        Replace values that are filtered out based on the Cooks distance with imputed
        values, and then re-run the whole DESeq2 pipeline on replaced values.
        """
        # Replace outlier counts
        self._replace_outliers()
        print(f"Refitting {sum(self.varm['replaced']) } outliers.\n")

        if sum(self.varm["replaced"]) > 0:
            # Refit dispersions and LFCs for genes that had outliers replaced
            self._refit_without_outliers()

    def _fit_MoM_dispersions(self):
        """ "Rough method of moments" initial dispersions fit.
        Estimates are the max of "robust" and "method of moments" estimates.
        """

        # Check that size_factors are available. If not, compute them.
        if "size_factors" not in self.obsm:
            self.fit_size_factors()

        non_zero = ~(self.X == 0).all(axis=0)

        rde = fit_rough_dispersions(
            self.X,
            self.obsm["size_factors"],
            self.obsm["design_matrix"],
        )
        mde = fit_moments_dispersions(self.X, self.obsm["size_factors"])
        alpha_hat = np.minimum(rde, mde)

        self.varm["_rough_dispersions"] = np.full(self.n_vars, np.NaN)
        self.varm["_rough_dispersions"][non_zero] = np.clip(
            alpha_hat, self.min_disp, self.max_disp
        )

    def _replace_outliers(self):
        """Replace values that are filtered out based
        on the Cooks distance with imputed values.
        """

        # Check that cooks distances are available. If not, compute them.
        if "cooks" not in self.layers:
            self.calculate_cooks()

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[1]

        # Check whether cohorts have enough samples to allow refitting
        n_or_more = (
            self.obsm["design_matrix"][
                self.obsm["design_matrix"].columns[-1]
            ].value_counts()
            >= self.min_replicates
        )
        # self.replaceable = pd.Series(
        #     n_or_more[  # TODO: cleanup
        #         self.obsm["design_matrix"][
        #             self.obsm["design_matrix"].columns[-1]
        #         ]
        #     ],
        #     index=self.obs_names,
        # )

        replaceable = n_or_more[  # TODO: could this be simplified?
            self.obsm["design_matrix"][self.obsm["design_matrix"].columns[-1]]
        ]

        self.obsm["replaceable"] = replaceable.values

        # Get positions of counts with cooks above threshold
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)
        idx = self.layers["cooks"] > cooks_cutoff
        self.varm["replaced"] = idx.any(axis=0)

        # Compute replacement counts: trimmed means * size_factors
        self.counts_to_refit = self[:, self.varm["replaced"]].copy()

        trim_base_mean = pd.DataFrame(
            trimmed_mean(
                self.counts_to_refit.X / self.obsm["size_factors"][:, None],
                trim=0.2,
                axis=0,
            ),
            index=self.counts_to_refit.var_names,
        )
        replacement_counts = (
            pd.DataFrame(
                trim_base_mean.values * self.obsm["size_factors"],
                index=self.counts_to_refit.var_names,
                columns=self.obs_names,
            )
            .astype(int)
            .T
        )

        if sum(self.varm["replaced"] > 0):
            self.counts_to_refit.X[
                idx[self.obsm["replaceable"]][:, self.varm["replaced"]]
            ] = replacement_counts.values[
                idx[self.obsm["replaceable"]][:, self.varm["replaced"]]
            ]
            # TODO : does this happen in place or not ?

    def _refit_without_outliers(
        self,
    ):
        """Re-run the whole DESeq2 pipeline with replaced outliers."""
        assert (
            self.refit_cooks
        ), "Trying to refit Cooks outliers but the 'refit_cooks' flag is set to False"

        # Check that replaced counts are available. If not, compute them.
        if not hasattr(self, "counts_to_refit"):
            self._replace_outliers()

        # Only refit genes for which replacing outliers hasn't resulted in all zeroes
        new_all_zeroes = (self.counts_to_refit.X == 0).all(axis=0)
        self.new_all_zeroes_genes = self.counts_to_refit.var_names[new_all_zeroes]
        self.counts_to_refit = self.counts_to_refit[:, ~new_all_zeroes].copy()

        sub_dds = DeseqDataSet(
            counts=pd.DataFrame(
                self.counts_to_refit.X,
                index=self.counts_to_refit.obs_names,
                columns=self.counts_to_refit.var_names,
            ),
            clinical=self.obs,
            design_factors=self.design_factors,
            min_mu=self.min_mu,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            refit_cooks=self.refit_cooks,
            min_replicates=self.min_replicates,
            beta_tol=self.beta_tol,
            n_cpus=self.n_processes,
            batch_size=self.batch_size,
        )

        # Use the same size factors
        sub_dds.obsm["size_factors"] = self.obsm["size_factors"]

        # Estimate gene-wise dispersions.
        sub_dds.fit_genewise_dispersions()

        # Compute trend dispersions.
        # Note: the trend curve is not refitted.
        sub_dds.uns["trend_coeffs"] = self.uns["trend_coeffs"]
        sub_dds.varm["_normed_means"] = self.counts_to_refit.varm["_normed_means"]
        sub_dds.varm["fitted_dispersions"] = dispersion_trend(
            sub_dds.varm["_normed_means"],
            sub_dds.uns["trend_coeffs"],
        )

        # Estimate MAP dispersions.
        # Note: the prior variance is not recomputed.
        sub_dds.uns["_squared_logres"] = self.uns["_squared_logres"]
        sub_dds.uns["prior_disp_var"] = self.uns["prior_disp_var"]

        sub_dds.fit_MAP_dispersions()

        # Estimate log-fold changes (in natural log scale)
        sub_dds.fit_LFC()

        # Replace values in main object
        to_replace = self.varm["replaced"].copy()
        # Only replace if genes are not all zeroes after outlier replacement
        to_replace[to_replace] = ~new_all_zeroes  # TODO : is this inplace? (should be)

        self.varm["_normed_means"][to_replace] = sub_dds.varm["_normed_means"]
        self.varm["LFC"][to_replace] = sub_dds.varm["LFC"]
        self.varm["dispersions"][to_replace] = sub_dds.varm["dispersions"]
        self.layers["replace_cooks"] = self.layers["cooks"].copy()
        self[self.obsm["replaceable"], to_replace].layers["replace_cooks"].fill(0)

        # Take into account new all-zero genes
        self[:, self.counts_to_refit.var_names[new_all_zeroes]].varm[
            "_normed_means"
        ] = np.zeros(new_all_zeroes.sum())
        self[:, self.counts_to_refit.var_names[new_all_zeroes]].varm["LFC"] = np.zeros(
            new_all_zeroes.sum()
        )
