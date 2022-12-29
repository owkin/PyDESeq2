import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from scipy.optimize import root_scalar
from scipy.stats import f
from statsmodels.stats.multitest import multipletests

from pydeseq2.utils import get_num_processes
from pydeseq2.utils import nbinomGLM
from pydeseq2.utils import wald_test


class DeseqStats:
    """PyDESeq2 statistical tests for differential expression.

    Implements p-value estimation for differential gene expression according
    to the DESeq2 pipeline [1]_.

    Also supports apeGLM log-fold change shrinkage [2]_.

    Parameters
    ----------
    dds : DeseqDataSet
        DeseqDataSet for which dispersion and LFCs were already estimated.

    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05). (default: 0.05).

    cooks_filter : bool
        Whether to filter p-values based on cooks outliers. (default: True).

    independent_filter : bool
        Whether to perform independent filtering to correct p-value trends. (default: True).

    n_cpus : int
        Number of cpus to use for multiprocessing.
        If None, all available CPUs will be used. (default: None).

    prior_disp_var : ndarray
        Prior variance for LFCs, used for ridge regularization. (default: None).

    batch_size : int
        Number of tasks to allocate to each joblib parallel worker. (default: 128).

    joblib_verbosity : int
        The verbosity level for joblib tasks. The higher the value, the more updates
        are reported. (default: 0).

    Attributes
    ----------
    base_mean : pandas.Series
        Genewise means of normalized counts.

    LFCs : pandas.DataFrame
        Estimated log-fold change between conditions and intercept, in natural log scale.

    SE : pandas.Series
        Standard LFC error.

    statistics : pandas.Series
        Wald statistics.

    p_values : pandas.Series
        P-values estimated from Wald statistics.

    padj : pandas.Series
        P-values adjusted for multiple testing.

    results_df : pandas.DataFrame
        Summary of the statistical analysis.

    shrunk_LFCs : bool
        Whether LFCs are shrunk.

    n_processes : int
        Number of threads to use for multiprocessing.

    References
    ----------
    ..  [1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>
    ..  [2] Zhu, A., Ibrahim, J. G., & Love, M. I. (2019).
        "Heavy-tailed prior distributions for sequence count data:
        removing the noise and preserving large differences."
        Bioinformatics, 35(12), 2084-2092.
        <https://academic.oup.com/bioinformatics/article/35/12/2084/5159452>
    """  # noqa: E501

    def __init__(
        self,
        dds,
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
        n_cpus=None,
        prior_disp_var=None,
        batch_size=128,
        joblib_verbosity=0,
    ):
        assert hasattr(
            dds, "LFCs"
        ), "Please provide a fitted DeseqDataSet by first running the `deseq2` method."

        self.dds = dds
        self.alpha = alpha
        self.cooks_filter = cooks_filter
        self.independent_filter = independent_filter
        self.prior_disp_var = prior_disp_var
        self.base_mean = self.dds._normed_means
        self.LFCs = self.dds.LFCs.copy()
        # Set a flag to indicate that LFCs are unshrunk
        self.shrunk_LFCs = False
        self.n_processes = get_num_processes(n_cpus)
        self.batch_size = batch_size
        self.joblib_verbosity = joblib_verbosity

    def summary(self):
        """Run the statistical analysis and return the results in a DataFrame.

        The results are also stored in the `results_df` attribute.

        Returns
        -------
        pandas.DataFrame
            Estimated log fold changes, p-values and adjusted p-values for each gene.
        """

        if not hasattr(self, "p_values"):
            # Estimate p-values with Wald test
            self.run_wald_test()

        if self.cooks_filter:
            # Filter p-values based on Cooks outliers
            self._cooks_filtering()

        if not hasattr(self, "padj"):
            if self.independent_filter:
                # Compute adjusted p-values and correct p-value trend
                self._independent_filtering()
            else:
                # Compute adjusted p-values using the Benjamini-Hochberg method, without
                # correcting the p-value trend.
                self._p_value_adjustment()

        # Store the results in a DataFrame, in log2 scale for LFCs.
        self.results_df = pd.DataFrame(index=self.dds.dispersions.index)
        self.results_df["baseMean"] = self.base_mean
        self.results_df["log2FoldChange"] = self.LFCs[
            self.dds.design_matrix.columns[-1]
        ] / np.log(2)
        self.results_df["lfcSE"] = self.SE / np.log(2)
        self.results_df["stat"] = self.statistics
        self.results_df["pvalue"] = self.p_values
        self.results_df["padj"] = self.padj

        return self.results_df

    def lfc_shrink(self):
        """LFC shrinkage with an apeGLM prior [2]_.

        Shrinks LFCs using a heavy-tailed Cauchy prior, leaving p-values unchanged.

        Returns
        -------
        pandas.DataFrame or None
            If pvalues were already computed, return the results DataFrame with MAP LFCs,
            but unmodified stats and pvalues.
        """
        # Filter genes with all zero counts
        nonzero = self.dds.counts.sum(0) > 0
        counts_nonzero = self.dds.counts.loc[:, nonzero].values
        m = counts_nonzero.shape[1]

        size = (1.0 / self.dds.dispersions[nonzero]).values
        offset = np.log(self.dds.size_factors).values

        # Set priors
        prior_no_shrink_scale = 15
        prior_var = self._fit_prior_var()
        prior_scale = np.minimum(np.sqrt(prior_var), 1)

        X = self.dds.design_matrix.values

        print("Fitting MAP LFCs...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(nbinomGLM)(
                    X,
                    counts_nonzero[:, i],
                    size[i],
                    offset,
                    prior_no_shrink_scale,
                    prior_scale,
                )
                for i in range(m)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        lfcs, inv_hessians, l_bfgs_b_converged_ = zip(*res)

        self.LFCs.iloc[:, 1] = pd.Series(
            np.array(lfcs)[:, 1], index=self.dds.dispersions[nonzero].index
        )
        self.SE = pd.Series(
            np.array([np.sqrt(np.abs(inv_hess[1, 1])) for inv_hess in inv_hessians]),
            index=self.dds.dispersions[nonzero].index,
        )
        self._lcf_shrink_converged = pd.Series(
            np.array(l_bfgs_b_converged_), index=self.dds.dispersions[nonzero].index
        )

        # Set a flag to indicate that LFCs were shrunk
        self.shrunk_LFCs = True

        # Replace in results dataframe, if it exists
        if hasattr(self, "results_df"):
            self.results_df["log2FoldChange"] = self.LFCs[
                self.dds.design_matrix.columns[-1]
            ] / np.log(2)
            self.results_df["lfcSE"] = self.SE / np.log(2)
            return self.results_df

    def run_wald_test(self):
        """Perform a Wald test.

        Get gene-wise p-values for gene over/under-expression.`
        """

        n = len(self.LFCs)
        p = self.dds.design_matrix.shape[1]

        # Raise a warning if LFCs are shrunk.
        if self.shrunk_LFCs:
            print(
                "Note: running Wald test on shrunk LFCs. "
                "Some sequencing datasets show better performance with the testing "
                "separated from the use of the LFC prior."
            )

        # Get the LFC column corresponding to the design variable.
        idx = self.LFCs.columns.get_loc(self.dds.design_matrix.columns[-1])

        # Compute means according to the LFC model.
        mu = (
            np.exp(self.dds.design_matrix @ self.LFCs.T)
            .multiply(self.dds.size_factors, 0)
            .values
        )

        # Set regularization factors.
        if self.prior_disp_var is not None:
            D = np.diag(1 / self.prior_disp_var**2)
        else:
            D = np.diag(np.repeat(1e-6, p))

        X = self.dds.design_matrix.values
        disps = self.dds.dispersions.values
        LFCs = self.LFCs.values

        print("Running Wald tests...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(wald_test)(X, disps[i], LFCs[i], mu[:, i], D, idx)
                for i in range(n)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        pvals, stats, se = zip(*res)

        self.p_values = pd.Series(pvals, index=self.LFCs.index)
        self.statistics = pd.Series(stats, index=self.LFCs.index)
        self.SE = pd.Series(se, index=self.LFCs.index)

        # Account for possible all_zeroes due to outlier refitting in DESeqDataSet
        if self.dds.refit_cooks and self.dds.replaced.sum() > 0:

            self.SE.loc[self.dds.new_all_zeroes[self.dds.new_all_zeroes].index] = 0
            self.statistics.loc[
                self.dds.new_all_zeroes[self.dds.new_all_zeroes].index
            ] = 0
            self.p_values.loc[self.dds.new_all_zeroes[self.dds.new_all_zeroes].index] = 1

    def _independent_filtering(self):
        """Compute adjusted p-values using independent filtering.

        Corrects p-value trend (see [1]_)
        """

        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self.run_wald_test()

        lower_quantile = np.mean(self.base_mean == 0)

        if lower_quantile < 0.95:
            upper_quantile = 0.95
        else:
            upper_quantile = 1

        theta = np.linspace(lower_quantile, upper_quantile, 50)

        cutoffs = np.quantile(self.base_mean, theta)

        result = pd.DataFrame(
            np.nan, index=self.base_mean.index, columns=np.arange(len(theta))
        )

        for i, cutoff in enumerate(cutoffs):
            use = (self.base_mean >= cutoff) & (~self.p_values.isna())
            U2 = self.p_values[use]
            result.loc[use, i] = multipletests(U2, alpha=self.alpha, method="fdr_bh")[1]

        num_rej = (result < self.alpha).sum(0)
        lowess = sm.nonparametric.lowess(num_rej, theta, frac=1 / 5)

        if num_rej.max() <= 10:
            j = 0
        else:
            residual = num_rej[num_rej > 0] - lowess[num_rej > 0, 1]
            thresh = lowess[:, 1].max() - np.sqrt(np.mean(residual**2))

            if np.any(num_rej > thresh):
                j = np.where(num_rej > thresh)[0][0]
            else:
                j = 0

        self.padj = result.loc[:, j]

    def _p_value_adjustment(self):
        """Compute adjusted p-values using the Benjamini-Hochberg method.


        Does not correct the p-value trend.
        This method and the `_independent_filtering` are mutually exclusive.
        """
        if not hasattr(self, "p_values"):
            # Estimate p-values with Wald test
            self.run_wald_test()

        self.padj = pd.Series(np.nan, index=self.p_values.index)
        self.padj.loc[~self.p_values.isna()] = multipletests(
            self.p_values.dropna(), alpha=self.alpha, method="fdr_bh"
        )[1]

    def _cooks_filtering(self):
        """Filter p-values based on Cooks outliers."""

        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self.run_wald_test()

        m, p = self.dds.design_matrix.shape
        if p == 2:
            # If for a gene there are 3 samples or more that have more counts than the
            # maximum cooks sample, don't count this gene as an outlier.
            # Do this only if there are 2 cohorts.
            cooks_cutoff = f.ppf(0.99, p, m - p)

            # Check whether cohorts have enough samples to allow refitting
            # Only consider conditions with 3 or more samples (same as in R)
            n_or_more = (
                self.dds.design_matrix[self.dds.design_matrix.columns[-1]].value_counts()
                >= 3
            )
            use_for_max = pd.Series(
                n_or_more[self.dds.design_matrix[self.dds.design_matrix.columns[-1]]]
            )
            use_for_max.index = self.dds.design_matrix.index

            # Take into account whether we already replaced outliers
            if self.dds.refit_cooks and self.dds.replaced.sum() > 0:
                cooks_outlier = (
                    self.dds.replace_cooks.loc[:, use_for_max] > cooks_cutoff
                ).any(axis=1)

            else:
                cooks_outlier = (self.dds.cooks.loc[:, use_for_max] > cooks_cutoff).any(
                    axis=1
                )

            pos = self.dds.cooks[cooks_outlier].to_numpy().argmax(1)
            cooks_outlier.update(
                (
                    self.dds.counts.loc[:, cooks_outlier]
                    > self.dds.counts.loc[:, cooks_outlier].to_numpy()[
                        pos, np.arange(len(pos))
                    ]
                ).sum(0)
                < 3
            )
            self.p_values[cooks_outlier] = np.nan

    def _fit_prior_var(self, min_var=1e-6, max_var=400):
        """Estimate the prior variance of the apeGLM model.

        Returns shrinkage factors.

        Parameters
        ----------
        min_var : float
            Lower bound for prior variance. (default: 1e-6).

        max_var : float
            Upper bound for prior variance. (default: 400).

        Returns
        -------
        float
            Estimated prior variance.
        """
        keep = ~self.LFCs.iloc[:, 1].isna()
        S = self.LFCs[keep].iloc[:, 1] ** 2
        D = self.SE[keep] ** 2

        def objective(a):
            # Equation to solve
            coeff = 1 / (2 * (a + D) ** 2)
            return ((S - D) * coeff).sum() / coeff.sum() - a

        # The prior variance is the zero of the above function.
        if objective(min_var) < 0:
            return min_var
        else:
            return root_scalar(objective, bracket=[min_var, max_var]).root
