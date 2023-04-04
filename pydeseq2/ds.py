import time
from typing import List
from typing import Optional

# import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from IPython.display import display  # type: ignore
from joblib import Parallel  # type: ignore
from joblib import delayed  # type: ignore
from joblib import parallel_backend  # type: ignore
from scipy.optimize import root_scalar  # type: ignore
from scipy.stats import f  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore

from pydeseq2.dds import DeseqDataSet
from pydeseq2.utils import get_num_processes
from pydeseq2.utils import nbinomGLM
from pydeseq2.utils import wald_test


class DeseqStats:
    """PyDESeq2 statistical tests for differential expression.

    Implements p-value estimation for differential gene expression according
    to the DESeq2 pipeline :cite:p:`DeseqStats-love2014moderated`.

    Also supports apeGLM log-fold change shrinkage :cite:p:`DeseqStats-zhu2019heavy`.

    Parameters
    ----------
    dds : DeseqDataSet
        DeseqDataSet for which dispersion and LFCs were already estimated.

    contrast : list or None
        A list of three strings, in the following format:
        ``['variable_of_interest', 'tested_level', 'ref_level']``.
        Names must correspond to the clinical data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
        to 'condition A'. If None, the last variable from the design matrix is chosen
        as the variable of interest, and the reference level is picked alphabetically.
        (default: ``None``).

    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05).
        (default: ``0.05``).

    cooks_filter : bool
        Whether to filter p-values based on cooks outliers. (default: ``True``).

    independent_filter : bool
        Whether to perform independent filtering to correct p-value trends.
        (default: ``True``).

    n_cpus : int
        Number of cpus to use for multiprocessing.
        If None, all available CPUs will be used. (default: ``None``).

    prior_LFC_var : ndarray
        Prior variance for LFCs, used for ridge regularization. (default: ``None``).

    batch_size : int
        Number of tasks to allocate to each joblib parallel worker. (default: ``128``).

    joblib_verbosity : int
        The verbosity level for joblib tasks. The higher the value, the more updates
        are reported. (default: ``0``).

    Attributes
    ----------
    base_mean : pandas.Series
        Genewise means of normalized counts.

    contrast_vector : ndarray
        Vector encoding the contrast (variable being tested).

    contrast_idx : int
        Index of the LFC column corresponding to the variable being tested.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Depending on the contrast that is provided to the
        DeseqStats object, it may differ from the DeseqDataSet design matrix, as the
        reference level may need to be adapted.

    LFC : pandas.DataFrame
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
    .. bibliography::
        :keyprefix: DeseqStats-
    """

    def __init__(
        self,
        dds: DeseqDataSet,
        contrast: Optional[List[str]] = None,
        alpha: float = 0.05,
        cooks_filter: bool = True,
        independent_filter: bool = True,
        n_cpus: Optional[int] = None,
        prior_LFC_var: Optional[np.ndarray] = None,
        batch_size: int = 128,
        joblib_verbosity: int = 0,
    ) -> None:
        assert (
            "LFC" in dds.varm
        ), "Please provide a fitted DeseqDataSet by first running the `deseq2` method."

        self.dds = dds

        self.alpha = alpha
        self.cooks_filter = cooks_filter
        self.independent_filter = independent_filter
        self.base_mean = self.dds.varm["_normed_means"].copy()
        self.prior_LFC_var = prior_LFC_var

        # Check the validity of the contrast (if provided) or build it.
        self._build_contrast(contrast)

        # Initialize the design matrix and LFCs. If the chosen reference level are the
        # same as in dds, keep them unchanged. Otherwise, change reference level.
        self.design_matrix = self.dds.obsm["design_matrix"].copy()
        self.LFC = self.dds.varm["LFC"].copy()

        # Build a contrast vector corresponding to the variable and levels of interest
        self._build_contrast_vector()

        # Set a flag to indicate that LFCs are unshrunk
        self.shrunk_LFCs = False
        self.n_processes = get_num_processes(n_cpus)
        self.batch_size = batch_size
        self.joblib_verbosity = joblib_verbosity

        # If the `refit_cooks` attribute of the dds object is True, check that outliers
        # were actually refitted.
        if self.dds.refit_cooks and "replaced" not in self.dds.varm:
            raise AttributeError(
                "dds has 'refit_cooks' set to True but Cooks outliers have not been "
                "refitted. Please run 'dds.refit()' first or set 'dds.refit_cooks' "
                "to False."
            )

    def summary(self) -> None:
        """Run the statistical analysis.

        The results are stored in the ``results_df`` attribute.
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
        self.results_df = pd.DataFrame(index=self.dds.var_names)
        self.results_df["baseMean"] = self.base_mean
        self.results_df["log2FoldChange"] = self.LFC @ self.contrast_vector / np.log(2)
        self.results_df["lfcSE"] = self.SE / np.log(2)
        self.results_df["stat"] = self.statistics
        self.results_df["pvalue"] = self.p_values
        self.results_df["padj"] = self.padj

        print(
            f"Log2 fold change & Wald test p-value: "
            f"{self.contrast[0]} {self.contrast[1]} vs {self.contrast[2]}"
        )
        display(self.results_df)

    def run_wald_test(self) -> None:
        """Perform a Wald test.

        Get gene-wise p-values for gene over/under-expression.`
        """

        num_genes = self.dds.n_vars
        num_vars = self.design_matrix.shape[1]

        # Raise a warning if LFCs are shrunk.
        if self.shrunk_LFCs:
            print(
                "Note: running Wald test on shrunk LFCs. "
                "Some sequencing datasets show better performance with the testing "
                "separated from the use of the LFC prior."
            )

        mu = (
            np.exp(self.design_matrix @ self.LFC.T)
            .multiply(self.dds.obsm["size_factors"], 0)
            .values
        )

        # Set regularization factors.
        if self.prior_LFC_var is not None:
            ridge_factor = np.diag(1 / self.prior_LFC_var**2)
        else:
            ridge_factor = np.diag(np.repeat(1e-6, num_vars))

        design_matrix = self.design_matrix.values
        LFCs = self.LFC.values

        print("Running Wald tests...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(wald_test)(
                    design_matrix=design_matrix,
                    disp=self.dds.varm["dispersions"][i],
                    lfc=LFCs[i],
                    mu=mu[:, i],
                    ridge_factor=ridge_factor,
                    contrast=self.contrast_vector,
                )
                for i in range(num_genes)
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        pvals, stats, se = zip(*res)

        self.p_values: pd.Series = pd.Series(pvals, index=self.dds.var_names)
        self.statistics: pd.Series = pd.Series(stats, index=self.dds.var_names)
        self.SE: pd.Series = pd.Series(se, index=self.dds.var_names)

        # Account for possible all_zeroes due to outlier refitting in DESeqDataSet
        if self.dds.refit_cooks and self.dds.varm["replaced"].sum() > 0:
            self.SE.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.statistics.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.p_values.loc[self.dds.new_all_zeroes_genes] = 1.0

    def lfc_shrink(self, coeff: str) -> None:
        """LFC shrinkage with an apeGLM prior :cite:p:`DeseqStats-zhu2019heavy`.

        Shrinks LFCs using a heavy-tailed Cauchy prior, leaving p-values unchanged.

        Parameters
        ----------
        coeff : str
            The LFC coefficient to shrink.
            If the desired coefficient is not available, it may be set from the
            :class:`pydeseq2.dds.DeseqDataSet` argument ``ref_level``.
        """

        if coeff not in self.LFC.columns:
            raise KeyError(
                f"The coeff argument ('{coeff}') should be one the LFC columns. "
                f"If not available,  it can be set from DeseqDataSet's `ref_level` "
                f"argument."
            )

        coeff_idx = self.LFC.columns.get_loc(coeff)

        size = 1.0 / self.dds.varm["dispersions"]
        offset = np.log(self.dds.obsm["size_factors"])

        # Set priors
        prior_no_shrink_scale = 15
        prior_var = self._fit_prior_var(coeff_idx=coeff_idx)
        prior_scale = np.minimum(np.sqrt(prior_var), 1)

        design_matrix = self.design_matrix.values

        print("Fitting MAP LFCs...")
        start = time.time()
        with parallel_backend("loky", inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_processes,
                verbose=self.joblib_verbosity,
                batch_size=self.batch_size,
            )(
                delayed(nbinomGLM)(
                    design_matrix=design_matrix,
                    counts=self.dds.X[:, i],
                    size=size[i],
                    offset=offset,
                    prior_no_shrink_scale=prior_no_shrink_scale,
                    prior_scale=prior_scale,
                    optimizer="L-BFGS-B",
                    shrink_index=coeff_idx,
                )
                for i in self.dds.non_zero_idx
            )
        end = time.time()
        print(f"... done in {end-start:.2f} seconds.\n")

        lfcs, inv_hessians, l_bfgs_b_converged_ = zip(*res)

        self.LFC.iloc[:, coeff_idx].update(
            pd.Series(
                np.array(lfcs)[:, coeff_idx],
                index=self.dds.non_zero_genes,
            )
        )

        self.SE.update(
            pd.Series(
                np.array(
                    [
                        np.sqrt(np.abs(inv_hess[coeff_idx, coeff_idx]))
                        for inv_hess in inv_hessians
                    ]
                ),
                index=self.dds.non_zero_genes,
            )
        )

        self._LFC_shrink_converged = pd.Series(np.NaN, index=self.dds.var_names)
        self._LFC_shrink_converged.update(
            pd.Series(l_bfgs_b_converged_, index=self.dds.non_zero_genes)
        )

        # Set a flag to indicate that LFCs were shrunk
        self.shrunk_LFCs = True

        # Replace in results dataframe, if it exists
        if hasattr(self, "results_df"):
            self.results_df["log2FoldChange"] = self.LFC.iloc[:, coeff_idx] / np.log(2)
            self.results_df["lfcSE"] = self.SE / np.log(2)

            print(
                f"Shrunk Log2 fold change & Wald test p-value: "
                f"{self.contrast[0]} {self.contrast[1]} vs {self.contrast[2]}"
            )

            display(self.results_df)

    def _independent_filtering(self) -> None:
        """Compute adjusted p-values using independent filtering.

        Corrects p-value trend (see :cite:p:`DeseqStats-love2014moderated`)
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
            np.nan, index=self.dds.var_names, columns=np.arange(len(theta))
        )

        for i, cutoff in enumerate(cutoffs):
            use = (self.base_mean >= cutoff) & (~self.p_values.isna())
            U2 = self.p_values[use]
            if not U2.empty:
                result.loc[use, i] = multipletests(
                    U2, alpha=self.alpha, method="fdr_bh"
                )[1]

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

    def _p_value_adjustment(self) -> None:
        """Compute adjusted p-values using the Benjamini-Hochberg method.

        Does not correct the p-value trend.
        This method and the `_independent_filtering` are mutually exclusive.
        """
        if not hasattr(self, "p_values"):
            # Estimate p-values with Wald test
            self.run_wald_test()

        self.padj = pd.Series(np.nan, index=self.dds.var_names)
        self.padj.loc[~self.p_values.isna()] = multipletests(
            self.p_values.dropna(), alpha=self.alpha, method="fdr_bh"
        )[1]

    def _cooks_filtering(self) -> None:
        """Filter p-values based on Cooks outliers."""

        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self.run_wald_test()

        num_samples = self.dds.n_obs
        num_vars = self.design_matrix.shape[-1]
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)

        # If for a gene there are 3 samples or more that have more counts than the
        # maximum cooks sample, don't count this gene as an outlier.
        # Do this only if there are 2 cohorts.
        if num_vars == 2:
            # Check whether cohorts have enough samples to allow refitting
            # Only consider conditions with 3 or more samples (same as in R)
            n_or_more = self.design_matrix.iloc[:, self.contrast_idx].value_counts() >= 3
            use_for_max = pd.Series(
                n_or_more[self.design_matrix.iloc[:, self.contrast_idx]]
            )
            use_for_max.index = self.dds.obs_names

        else:
            use_for_max = pd.Series(True, index=self.dds.obs_names)

        # Take into account whether we already replaced outliers
        if self.dds.refit_cooks and self.dds.varm["replaced"].sum() > 0:
            cooks_outlier = (
                (self.dds[use_for_max, :].layers["replace_cooks"] > cooks_cutoff)
                .any(axis=0)
                .copy()
            )

        else:
            cooks_outlier = (
                (self.dds[use_for_max, :].layers["cooks"] > cooks_cutoff)
                .any(axis=0)
                .copy()
            )

        pos = self.dds[:, cooks_outlier].layers["cooks"].argmax(0)

        cooks_outlier[cooks_outlier] = (
            self.dds[:, cooks_outlier].X
            > self.dds[:, cooks_outlier].X[pos, np.arange(len(pos))]
        ).sum(0) < 3

        self.p_values[cooks_outlier] = np.nan

    def _fit_prior_var(
        self, coeff_idx: str, min_var: float = 1e-6, max_var: float = 400.0
    ) -> float:
        """Estimate the prior variance of the apeGLM model.

        Returns shrinkage factors.

        Parameters
        ----------
        coeff_idx : str
            Index of the coefficient to shrink.

        min_var : float
            Lower bound for prior variance. (default: ``1e-6``).

        max_var : float
            Upper bound for prior variance. (default: ``400``).

        Returns
        -------
        float
            Estimated prior variance.
        """

        keep = ~self.LFC.iloc[:, coeff_idx].isna()
        S = self.LFC[keep].iloc[:, coeff_idx] ** 2
        D = self.SE[keep] ** 2

        def objective(a: float) -> float:
            # Equation to solve
            coeff = 1 / (2 * (a + D) ** 2)
            return ((S - D) * coeff).sum() / coeff.sum() - a

        # The prior variance is the zero of the above function.
        if objective(min_var) < 0:
            return min_var
        else:
            return root_scalar(objective, bracket=[min_var, max_var]).root

    def _build_contrast(self, contrast: Optional[List[str]] = None) -> None:
        """
        Check the validity of the contrast (if provided). If not, build a default
        contrast, corresponding to the last column of the design matrix.

        A contrast should be a list of three strings, in the following format:
        ``['variable_of_interest', 'tested_level', 'reference_level']``.
        Names must correspond to the clinical data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B'
        compared to 'condition A'. If None, the last variable from the design matrix
        is chosen as the variable of interest, and the reference level is picked
        alphabetically.

        Parameters
        ----------
        contrast : list or None
            A list of three strings, in the following format:
            ``['variable_of_interest', 'tested_level', 'reference_level']``.
            (default: ``None``).
        """

        if contrast is not None:  # Test contrast if provided
            if len(contrast) != 3:
                raise ValueError("The contrast should contain three strings.")
            if contrast[0] not in self.dds.design_factors:
                raise KeyError(
                    f"The contrast variable ('{contrast[0]}') should be one "
                    f"of the design factors."
                )
            if contrast[1] not in self.dds.obs[contrast[0]].values:
                raise KeyError(
                    f"The tested level ('{contrast[1]}') should correspond to one of the"
                    f" levels of '{contrast[0]}'"
                )
            if contrast[2] not in self.dds.obs[contrast[0]].values:
                raise KeyError(
                    f"The reference level ('{contrast[2]}') should correspond to one of"
                    f" the levels of '{contrast[0]}'"
                )
            self.contrast = contrast
        else:  # Build contrast if None
            factor = self.dds.design_factors[-1]
            factor_col = next(
                col
                for col in self.dds.obsm["design_matrix"].columns
                if col.startswith(factor)
            )
            split_col = factor_col.split("_")
            self.contrast = [split_col[0], split_col[1], split_col[-1]]

    def _build_contrast_vector(self) -> None:
        """
        Build a vector corresponding to the desired contrast.

        Allows to test any pair of levels without refitting LFCs.
        """
        factor = self.contrast[0]
        alternative = self.contrast[1]
        ref = self.contrast[2]
        contrast_level = f"{factor}_{alternative}_vs_{ref}"

        self.contrast_vector = np.zeros(self.LFC.shape[-1])
        if contrast_level in self.design_matrix.columns:
            self.contrast_idx = self.LFC.columns.get_loc(contrast_level)
            self.contrast_vector[self.contrast_idx] = 1
        elif f"{factor}_{ref}_vs_{alternative}" in self.design_matrix.columns:
            # Reference and alternative are inverted
            self.contrast_idx = self.LFC.columns.get_loc(
                f"{factor}_{ref}_vs_{alternative}"
            )
            self.contrast_vector[self.contrast_idx] = -1
        else:
            # Need to change reference
            # Get any column corresponding to the desired factor and extract old ref
            old_ref = next(
                col for col in self.LFC.columns if col.startswith(factor)
            ).split("_")[-1]
            new_alternative_idx = self.LFC.columns.get_loc(
                f"{factor}_{alternative}_vs_{old_ref}"
            )
            new_ref_idx = self.LFC.columns.get_loc(f"{factor}_{ref}_vs_{old_ref}")
            self.contrast_vector[new_alternative_idx] = 1
            self.contrast_vector[new_ref_idx] = -1
