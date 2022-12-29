import time
import warnings

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


class DeseqDataSet:
    r"""A class to implement dispersion and log fold-change (LFC) estimation.

    Follows the DESeq2 pipeline [1]_.

    Parameters
    ----------
    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.

    clinical : pandas.DataFrame
        DataFrame containing clinical information.
        Must be indexed by sample barcodes.

    design_factor : str
        Name of the column of clinical to be used as a design variable.
        (default: 'high_grade').

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
    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    n_processes : int
        Number of cpus to use for multiprocessing.

    size_factors : pandas.Series
        DESeq normalization factors.

    non_zero_genes : pandas.Index
        Index of genes that have non-uniformly zero counts.

    genewise_dispersions : pandas.Series
        Initial estimates of gene counts dispersions.

    trend_coeffs : pandas.Series
        Coefficients of the trend curve: :math:`f(\mu) = \alpha_1/ \mu + a_0`.

    fitted_dispersions : pandas.Series
        Genewise dispersions regressed on the trend curve.

    prior_disp_var : float
        Dispersion prior of genewise dispersions,
        used for dispersion shrinkage towards the trend curve.

    MAP_dispersions : pandas.Series
        MAP dispersions, after shrinkage towards the trend curve and before filtering.

    dispersions : pandas.Series
        Final dispersion estimates, after filtering MAP outliers.

    LFCs : pandas.DataFrame
        Log-fold change and intercept parameters, in natural log scale.

    cooks : pandas.DataFrame
        Cooks distances, used for outlier detection.

    replaceable : pandas.Series
        Whether counts are replaceable, i.e. if a given condition has enough samples.

    replaced : pandas.Series
        Counts which were replaced.

    replace_cooks : pandas.DataFrame
        Cooks distances after replacement.

    counts_to_refit : pandas.DataFrame
        Read counts after replacement,
        for which dispersions and LFCs must be fitted again.

    new_all_zeroes : pandas.Series
        Genes which have only zero counts after outlier replacement.

    _rough_dispersions : pandas.Series
        Intial method-of-moments estimates of the dispersions

    References
    ----------
    ..  [1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>
    """  # noqa: E501

    def __init__(
        self,
        counts,
        clinical,
        design_factor="high_grade",
        reference_level=None,
        min_mu=0.5,
        min_disp=1e-8,
        max_disp=10,
        refit_cooks=True,
        min_replicates=7,
        beta_tol=1e-8,
        n_cpus=None,
        batch_size=128,
        joblib_verbosity=0,
    ):
        """Initialize the DeseqDataSet instance, computing the design matrix and
        the number of multiprocessing threads.
        """

        # Test counts before going further
        test_valid_counts(counts)
        self.counts = counts

        # Import clinical data and convert design_column to string
        self.clinical = clinical
        if not self.counts.index.identical(self.clinical.index):
            raise ValueError(
                "The count matrix and clinical data should have the same index."
            )
        self.design_factor = design_factor
        if self.clinical[self.design_factor].isna().any():
            raise ValueError("NaNs are not allowed in the design factor.")
        self.clinical[self.design_factor] = self.clinical[self.design_factor].astype(str)

        # Build the design matrix (splits the dataset in two cohorts)
        self.design_matrix = build_design_matrix(
            self.clinical,
            design_factor,
            ref=reference_level,
            expanded=False,
            intercept=True,
        )
        self.min_mu = min_mu
        self.min_disp = min_disp
        self.max_disp = np.maximum(max_disp, len(self.counts))
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
        _, self.size_factors = deseq2_norm(self.counts)
        end = time.time()
        print(f"... done in {end - start:.2f} seconds.\n")

    def fit_genewise_dispersions(self):
        """Fit gene-wise dispersion estimates.

        Fits a negative binomial per gene, independently.
        """

        # Check that size factors are available. If not, compute them.
        if not hasattr(self, "size_factors"):
            self.fit_size_factors()

        # Finit init "method of moments" dispersion estimates
        self._fit_MoM_dispersions()

        # Exclude genes with all zeroes
        non_zero = ~(self.counts == 0).all()
        self.non_zero_genes = non_zero[non_zero].index
        counts_nonzero = self.counts.loc[:, self.non_zero_genes].values
        m = counts_nonzero.shape[1]

        # Convert design_matrix to numpy for speed
        X = self.design_matrix.values
        rough_disps = self._rough_dispersions.loc[self.non_zero_genes].values
        size_factors = self.size_factors.values

        with parallel_backend("loky", inner_max_num_threads=1):
            mu_hat_ = np.array(
                Parallel(
                    n_jobs=self.n_processes,
                    verbose=self.joblib_verbosity,
                    batch_size=self.batch_size,
                )(
                    delayed(fit_lin_mu)(
                        counts_nonzero[:, i],
                        size_factors,
                        X,
                        self.min_mu,
                    )
                    for i in range(m)
                )
            )

        self._mu_hat = pd.DataFrame(
            mu_hat_.T,
            index=self.counts.index,
            columns=self.non_zero_genes,
        )

        print("Fitting dispersions...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=self.n_processes, batch_size=self.batch_size)(
                delayed(fit_alpha_mle)(
                    counts_nonzero[:, i],
                    X,
                    mu_hat_[i, :],
                    rough_disps[i],
                    self.min_disp,
                    self.max_disp,
                )
                for i in range(m)
            )
        end = time.time()
        print(f"... done in {end - start:.2f} seconds.\n")

        dispersions_, l_bfgs_b_converged_ = zip(*res)
        self.genewise_dispersions = pd.Series(np.NaN, index=self.counts.columns)
        self.genewise_dispersions.update(
            pd.Series(dispersions_, index=self.non_zero_genes).clip(
                self.min_disp, self.max_disp
            )
        )
        self._genewise_converged = pd.Series(
            l_bfgs_b_converged_, index=self.non_zero_genes
        )

    def fit_dispersion_trend(self):
        r"""Fit the dispersion trend coefficients.

        .. math:: f(\mu) = \alpha_1/\mu + a_0.
        """

        # Check that genewise dispersions are available. If not, compute them.
        if not hasattr(self, "genewise_dispersions"):
            self.fit_genewise_dispersions()

        print("Fitting dispersion trend curve...")
        start = time.time()
        self._normed_means = self.counts.div(self.size_factors, 0).mean(0)

        # Exclude all-zero counts
        targets = self.genewise_dispersions.loc[self.non_zero_genes].copy()
        covariates = sm.add_constant(1 / self._normed_means.loc[self.non_zero_genes])

        for gene in targets.index:
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
            pred_ratios = self.genewise_dispersions.loc[covariates.index] / predictions

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

        self.trend_coeffs = pd.Series(coeffs, index=["a0", "a1"])
        self.fitted_dispersions = pd.Series(
            np.NaN, index=self.genewise_dispersions.index
        )
        self.fitted_dispersions.update(
            dispersion_trend(
                self._normed_means.loc[self.non_zero_genes], self.trend_coeffs
            )
        )

    def fit_dispersion_prior(self):
        """Fit dispersion variance priors and standard deviation of log-residuals.

        The computation is based on genes whose dispersions are above 100 * min_disp.
        """

        # Check that the dispersion trend curve was fitted. If not, fit it.
        if not hasattr(self, "trend_coeffs"):
            self.fit_dispersion_trend()

        # Exclude genes with all zeroes
        m = len(self.non_zero_genes)
        p = self.design_matrix.shape[1]

        # Fit dispersions to the curve, and compute log residuals
        disp_residuals = np.log(self.genewise_dispersions) - np.log(
            self.fitted_dispersions
        )

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = self.genewise_dispersions.loc[self.non_zero_genes] >= (
            100 * self.min_disp
        )
        self._squared_logres = np.abs(
            disp_residuals.loc[self.non_zero_genes][above_min_disp]
        ).median() ** 2 / norm.ppf(0.75)
        self.prior_disp_var = np.maximum(
            self._squared_logres - polygamma(1, (m - p) / 2), 0.25
        )

    def fit_MAP_dispersions(self):
        """Fit Maximum a Posteriori dispersion estimates.

        After MAP dispersions are fit, filter genes for which we don't apply shrinkage.
        """

        # Check that the dispersion prior variance is available. If not, compute it.
        if not hasattr(self, "prior_disp_var"):
            self.fit_dispersion_prior()

        # Exclude genes with all zeroes
        counts_nonzero = self.counts.loc[:, self.non_zero_genes].values
        m = counts_nonzero.shape[1]

        X = self.design_matrix.values
        mu_hat = self._mu_hat.values
        fit_disps = self.fitted_dispersions.loc[self.non_zero_genes].values

        print("Fitting MAP dispersions...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(fit_alpha_mle)(
                    counts_nonzero[:, i],
                    X,
                    mu_hat[:, i],
                    fit_disps[i],
                    self.min_disp,
                    self.max_disp,
                    self.prior_disp_var,
                    True,
                    True,
                )
                for i in range(m)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        dispersions_, l_bfgs_b_converged_ = zip(*res)
        self.MAP_dispersions = pd.Series(np.NaN, index=self.fitted_dispersions.index)
        self.MAP_dispersions.update(
            pd.Series(dispersions_, index=self.non_zero_genes).clip(
                self.min_disp, self.max_disp
            )
        )
        self._MAP_converged = pd.Series(l_bfgs_b_converged_, index=self.non_zero_genes)

        # Filter outlier genes for which we won't apply shrinkage
        self.dispersions = self.MAP_dispersions.copy()
        self._outlier_genes = np.log(self.dispersions) > np.log(
            self.fitted_dispersions
        ) + 2 * np.sqrt(self._squared_logres)
        self.dispersions[self._outlier_genes] = self.genewise_dispersions[
            self._outlier_genes
        ]

    def fit_LFC(self):
        """Fit log fold change (LFC) coefficients.

        In the 2-level setting, the intercept corresponds to the base mean,
        while the second is the actual LFC coefficient, in natural log scale.
        """

        # Check that MAP dispersions are available. If not, compute them.
        if not hasattr(self, "dispersions"):
            self.fit_MAP_dispersions()

        # Exclude genes with all zeroes
        counts_nonzero = self.counts.loc[:, self.non_zero_genes].values
        m = counts_nonzero.shape[1]

        X = self.design_matrix.values
        size_factors = self.size_factors.values
        disps = self.dispersions.loc[self.non_zero_genes].values

        print("Fitting LFCs...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(irls_solver)(
                    counts_nonzero[:, i],
                    size_factors,
                    X,
                    disps[i],
                    self.min_mu,
                    self.beta_tol,
                )
                for i in range(m)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        MAP_lfcs_, mu_, hat_diagonals_, converged_ = zip(*res)

        self.LFCs = pd.DataFrame(
            np.NaN, index=self.dispersions.index, columns=self.design_matrix.columns
        )

        self.LFCs.update(
            pd.DataFrame(
                MAP_lfcs_, index=self.non_zero_genes, columns=self.design_matrix.columns
            )
        )

        self._mu_LFC = pd.DataFrame(
            np.array(mu_).T, index=self.counts.index, columns=self.non_zero_genes
        )

        self._hat_diagonals = pd.DataFrame(
            np.NaN,
            index=self.dispersions.index,
            columns=self.design_matrix.index,
        ).T

        self._hat_diagonals.update(
            pd.DataFrame(
                hat_diagonals_,
                index=self.non_zero_genes,
                columns=self.design_matrix.index,
            ).T
        )

        self._LFC_converged = pd.DataFrame(converged_, index=self.non_zero_genes)

    def calculate_cooks(self):
        """Compute Cook's distance for outlier detection.

        Measures the contribution of a single entry to the output of LFC estimation.
        """

        # Check that MAP dispersions are available. If not, compute them.
        if not hasattr(self, "dispersions"):
            self.fit_MAP_dispersions()

        p = self.design_matrix.shape[1]
        # Keep only non-zero genes
        normed_counts = self.counts.loc[:, self.non_zero_genes].div(self.size_factors, 0)
        dispersions = robust_method_of_moments_disp(normed_counts, self.design_matrix)
        V = self._mu_LFC + dispersions * self._mu_LFC**2
        squared_pearson_res = (
            self.counts.loc[:, self.non_zero_genes] - self._mu_LFC
        ) ** 2 / V
        self.cooks = (
            squared_pearson_res
            / p
            * (self._hat_diagonals / (1 - self._hat_diagonals) ** 2)
        ).T

    def refit(self):
        """Refit Cook outliers.

        Replace values that are filtered out based on the Cooks distance with imputed
        values, and then re-run the whole DESeq2 pipeline on replaced values.
        """
        # Replace outlier counts
        self._replace_outliers()
        print(f"Refitting {self.replaced.sum()} outliers.\n")

        if self.replaced.sum() > 0:
            # Refit dispersions and LFCs for genes that had outliers replaced
            self._refit_without_outliers()

    def _fit_MoM_dispersions(self):
        """ "Rough method of moments" initial dispersions fit.
        Estimates are the max of "robust" and "method of moments" estimates.
        """

        # Check that size_factors are available. If not, compute them.
        if not hasattr(self, "size_factors"):
            self.fit_size_factors()

        rde = fit_rough_dispersions(self.counts, self.size_factors, self.design_matrix)
        mde = fit_moments_dispersions(self.counts, self.size_factors)
        alpha_hat = np.minimum(rde, mde)
        self._rough_dispersions = np.clip(alpha_hat, self.min_disp, self.max_disp)

    def _replace_outliers(self):
        """Replace values that are filtered out based
        on the Cooks distance with imputed values.
        """

        # Check that cooks distances are available. If not, compute them.
        if not hasattr(self, "cooks"):
            self.calculate_cooks()

        m, p = self.design_matrix.shape

        # Check whether cohorts have enough samples to allow refitting
        n_or_more = (
            self.design_matrix[self.design_matrix.columns[-1]].value_counts()
            >= self.min_replicates
        )
        self.replaceable = pd.Series(
            n_or_more[self.design_matrix[self.design_matrix.columns[-1]]]
        )
        self.replaceable.index = self.design_matrix.index

        # Get positions of counts with cooks above threshold
        cooks_cutoff = f.ppf(0.99, p, m - p)
        idx = (self.cooks > cooks_cutoff).T
        self.replaced = idx.any(axis=0)

        # Compute replacement counts: trimmed means * size_factors
        self.counts_to_refit = self.counts.loc[:, self.replaced].copy()

        trim_base_mean = pd.DataFrame(
            trimmed_mean(
                self.counts_to_refit.divide(self.size_factors, 0),
                trim=0.2,
                axis=0,
            ),
            index=self.counts_to_refit.columns,
        )
        replacement_counts = (
            pd.DataFrame(
                trim_base_mean.values * self.size_factors.values,
                index=self.counts_to_refit.columns,
                columns=self.size_factors.index,
            )
            .astype(int)
            .T
        )

        self.counts_to_refit[
            idx[self.replaceable].loc[:, self.replaced]
        ] = replacement_counts[idx[self.replaceable].loc[:, self.replaced]]

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
        self.new_all_zeroes = (self.counts_to_refit == 0).all()
        self.counts_to_refit = self.counts_to_refit.loc[:, ~self.new_all_zeroes]

        sub_dds = DeseqDataSet(
            counts=self.counts_to_refit,
            clinical=self.clinical,
            design_factor=self.design_factor,
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
        sub_dds.size_factors = self.size_factors

        # Estimate gene-wise dispersions.
        sub_dds.fit_genewise_dispersions()

        # Compute trend dispersions.
        # Note: the trend curve is not refitted.
        sub_dds.trend_coeffs = self.trend_coeffs
        sub_dds._normed_means = self.counts_to_refit.divide(self.size_factors, 0).mean(0)
        sub_dds.fitted_dispersions = dispersion_trend(
            sub_dds._normed_means,
            sub_dds.trend_coeffs,
        )

        # Estimate MAP dispersions.
        # Note: the prior variance is not recomputed.
        sub_dds._squared_logres = self._squared_logres
        sub_dds.prior_disp_var = self.prior_disp_var

        sub_dds.fit_MAP_dispersions()

        # Estimate log-fold changes (in natural log scale)
        sub_dds.fit_LFC()

        # Replace values in main object
        self._normed_means[
            self.replaced & (~self.new_all_zeroes)
        ] = sub_dds._normed_means
        self.LFCs[self.replaced & (~self.new_all_zeroes)] = sub_dds.LFCs
        self.dispersions[self.replaced & (~self.new_all_zeroes)] = sub_dds.dispersions
        self.replace_cooks = self.cooks.copy()
        self.replace_cooks.loc[
            self.replaced & (~self.new_all_zeroes), self.replaceable
        ] = 0

        # Take into account new all-zero genes
        self._normed_means.loc[self.new_all_zeroes[self.new_all_zeroes].index] = 0
        self.LFCs.loc[self.new_all_zeroes[self.new_all_zeroes].index] = 0
