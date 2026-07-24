import sys
import time
import warnings
from typing import Literal

# import anndata as ad
import numpy as np
import pandas as pd
from formulaic_contrasts import FormulaicContrasts
from scipy.optimize import root_scalar  # type: ignore
from scipy.stats import chi2  # type: ignore
from scipy.stats import false_discovery_control  # type: ignore

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.inference import Inference
from pydeseq2.utils import lowess
from pydeseq2.utils import make_MA_plot
from pydeseq2.utils import nb_nll


class DeseqStats:
    """PyDESeq2 statistical tests for differential expression.

    Implements p-value estimation for differential gene expression according
    to the DESeq2 pipeline :cite:p:`DeseqStats-love2014moderated`.

    Also supports apeGLM log-fold change shrinkage :cite:p:`DeseqStats-zhu2019heavy`.

    Parameters
    ----------
    dds : DeseqDataSet
        DeseqDataSet for which dispersion and LFCs were already estimated.

    test : str
        The statistical test to use for p-value estimation. One of ``["Wald", "LRT"]``.
        The Wald test assesses the significance of a single coefficient (or contrast),
        while the likelihood ratio test (``"LRT"``) compares the full model to a nested
        ``reduced`` model, as in R DESeq2's ``DESeq(dds, test="LRT", reduced=...)``.
        (default: ``"Wald"``).

    reduced : str, ndarray, pandas.DataFrame, optional
        The reduced model to compare against the full model when ``test="LRT"``.
        Either a formulaic formula string (e.g. ``"~group"``, which must be nested in the
        design of ``dds``), or an explicit design matrix (as a numpy array or a pandas
        DataFrame whose columns are a subset of the full design matrix columns).
        Required when ``test="LRT"``, and must be left to ``None`` for the Wald test.
        (default: ``None``).

    contrast : list or ndarray
        Either a list of three strings or a numpy array.
        If a list of three strings, it must be in the following format:
        ``['variable_of_interest', 'tested_level', 'ref_level']``.
        Names must correspond to the metadata data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
        to 'condition A'.
        If a numpy array, it must be a contrast vector of the same length as the design
        matrix.

    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05).
        (default: ``0.05``).

    cooks_filter : bool
        Whether to filter p-values based on cooks outliers. (default: ``True``).

    independent_filter : bool
        Whether to perform independent filtering to correct p-value trends.
        (default: ``True``).

    prior_LFC_var : ndarray
        Prior variance for LFCs, used for ridge regularization. (default: ``None``).

    lfc_null : float
        The (log2) log fold change under the null hypothesis. (default: ``0``).

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values. By default, the normal
        Wald test assesses deviation of the estimated log fold change from the null
        hypothesis, as given by ``lfc_null``.
        One of ``["greaterAbs", "lessAbs", "greater", "less"]`` or ``None``.
        The alternative hypothesis corresponds to what the user wants to find rather
        than the null hypothesis. (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).

    quiet : bool
        Suppress deseq2 status updates during fit.

    Attributes
    ----------
    base_mean : pandas.Series
        Genewise means of normalized counts.

    lfc_null : float
        The (log2) log fold change under the null hypothesis.

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values.

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
        Wald statistics (``test="Wald"``) or likelihood ratio statistics
        (``test="LRT"``).

    p_values : pandas.Series
        P-values estimated from the Wald statistics (``test="Wald"``) or from the
        likelihood ratio statistics using a chi-squared distribution (``test="LRT"``).

    test : str
        The statistical test used for p-value estimation, one of ``["Wald", "LRT"]``.

    reduced_design_matrix : pandas.DataFrame, optional
        The reduced-model design matrix used for the likelihood ratio test
        (``None`` for the Wald test).

    padj : pandas.Series
        P-values adjusted for multiple testing.

    results_df : pandas.DataFrame
        Summary of the statistical analysis.

    shrunk_LFCs : bool
        Whether LFCs are shrunk.

    n_processes : int
        Number of threads to use for multiprocessing.

    quiet : bool
        Suppress deseq2 status updates during fit.

    References
    ----------
    .. bibliography::
        :keyprefix: DeseqStats-
    """

    def __init__(
        self,
        dds: DeseqDataSet,
        contrast: list[str] | np.ndarray,
        test: Literal["Wald", "LRT"] = "Wald",
        reduced: str | np.ndarray | pd.DataFrame | None = None,
        alpha: float = 0.05,
        cooks_filter: bool = True,
        independent_filter: bool = True,
        prior_LFC_var: np.ndarray | None = None,
        lfc_null: float = 0.0,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
        inference: Inference | None = None,
        quiet: bool = False,
        n_cpus: int | None = None,
    ) -> None:
        assert "LFC" in dds.varm, (
            "Please provide a fitted DeseqDataSet by first running the `deseq2` method."
        )

        self.dds = dds

        if test not in ("Wald", "LRT"):
            raise ValueError(f"test must be one of 'Wald' or 'LRT', got '{test}'.")
        self.test = test

        self.alpha = alpha
        self.cooks_filter = cooks_filter
        self.independent_filter = independent_filter
        self.base_mean = self.dds.var["_normed_means"].copy()
        self.prior_LFC_var = prior_LFC_var

        if lfc_null < 0 and alt_hypothesis in {"greaterAbs", "lessAbs"}:
            raise ValueError(
                f"The alternative hypothesis being {alt_hypothesis}, please provide a",
                f"positive lfc_null value (got {lfc_null}).",
            )
        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        # Initialize the design matrix and LFCs. If the chosen reference level are the
        # same as in dds, keep them unchanged. Otherwise, change reference level.
        self.design_matrix = self.dds.obsm["design_matrix"].copy()
        self.LFC = self.dds.varm["LFC"].copy()

        # Build the reduced-model design matrix for the likelihood ratio test.
        self.reduced_design_matrix = self._build_reduced_design_matrix(reduced)

        # Check the validity of the contrast (if provided) or build it.
        self.contrast: list[str] | np.ndarray
        if contrast is None:
            raise ValueError(
                """Default contrasts are no longer supported.
                The "contrast" argument must be provided."""
            )
        elif isinstance(contrast, np.ndarray):
            if contrast.shape[0] != self.dds.obsm["design_matrix"].shape[1]:
                raise ValueError(
                    "The contrast vector must have the same length as the design matrix."
                )
            self.contrast = contrast
            self.contrast_vector = contrast
        else:
            self.contrast = contrast
            self._build_contrast_vector()

        # Set a flag to indicate that LFCs are unshrunk
        self.shrunk_LFCs = False
        self.quiet = quiet

        if inference:
            if n_cpus:
                if hasattr(inference, "n_cpus"):
                    inference.n_cpus = n_cpus
                else:
                    warnings.warn(
                        "The provided inference object does not have an n_cpus "
                        "attribute, cannot override `n_cpus`.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Initialize the inference object.
        self.inference = inference or DefaultInference(n_cpus=n_cpus)

        # If the `refit_cooks` attribute of the dds object is True, check that outliers
        # were actually refitted.
        if self.dds.refit_cooks and "replaced" not in self.dds.var:
            raise AttributeError(
                "dds has 'refit_cooks' set to True but Cooks outliers have not been "
                "refitted. Please run 'dds.refit()' first or set 'dds.refit_cooks' "
                "to False."
            )

    @property
    def variables(self):
        """Get the names of the variables used in the model definition."""
        return self.dds.variables

    def summary(
        self,
        **kwargs,
    ) -> None:
        """Run the statistical analysis.

        The results are stored in the ``results_df`` attribute.

        Parameters
        ----------
        **kwargs
            Keyword arguments: providing new values for ``lfc_null`` or
            ``alt_hypothesis`` will override the corresponding ``DeseqStat`` attributes.
        """
        new_lfc_null = kwargs.get("lfc_null", "default")
        new_alt_hypothesis = kwargs.get("alt_hypothesis", "default")

        rerun_summary = False
        if new_lfc_null == "default":
            lfc_null = self.lfc_null
        else:
            lfc_null = new_lfc_null
        if new_alt_hypothesis == "default":
            alt_hypothesis = self.alt_hypothesis
        else:
            alt_hypothesis = new_alt_hypothesis
        if lfc_null < 0 and alt_hypothesis in {"greaterAbs", "lessAbs"}:
            raise ValueError(
                f"The alternative hypothesis being {alt_hypothesis}, please provide a",
                f"positive lfc_null value (got {lfc_null}).",
            )

        if self.test == "LRT" and (
            new_lfc_null != "default" or new_alt_hypothesis != "default"
        ):
            warnings.warn(
                "`lfc_null` and `alt_hypothesis` are only supported for the Wald test "
                "and are ignored when `test='LRT'`.",
                UserWarning,
                stacklevel=2,
            )

        if (
            not hasattr(self, "p_values")
            or self.lfc_null != lfc_null
            or self.alt_hypothesis != alt_hypothesis
        ):
            # Estimate p-values with the Wald test or the likelihood ratio test
            self.lfc_null = lfc_null
            self.alt_hypothesis = alt_hypothesis
            rerun_summary = True
            if self.test == "Wald":
                self.run_wald_test()
            else:
                self.run_likelihood_ratio_test()

        if self.cooks_filter:
            # Filter p-values based on Cooks outliers
            self._cooks_filtering()

        if not hasattr(self, "padj") or rerun_summary:
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

        if not self.quiet:
            pval_desc = (
                "Wald test p-value"
                if self.test == "Wald"
                else "likelihood ratio test p-value"
            )
            if isinstance(self.contrast, np.ndarray):
                # The contrast vector was directly provided
                print(
                    f"Log2 fold change & {pval_desc}, contrast vector: {self.contrast}"
                )
            else:
                # The factor is categorical
                print(
                    f"Log2 fold change & {pval_desc}: "
                    f"{self.contrast[0]} {self.contrast[1]} vs {self.contrast[2]}"
                )
            print(self.results_df)

    def run_wald_test(self) -> None:
        """Perform a Wald test.

        Get gene-wise p-values for gene over/under-expression.
        """
        num_vars = self.design_matrix.shape[1]

        # Raise a warning if LFCs are shrunk.
        if self.shrunk_LFCs:
            if not self.quiet:
                print(
                    "Note: running Wald test on shrunk LFCs. "
                    "Some sequencing datasets show better performance with the testing "
                    "separated from the use of the LFC prior.",
                    file=sys.stderr,
                )

        mu = (
            np.exp(self.design_matrix @ self.LFC.T)
            .multiply(self.dds.obs["size_factors"], 0)
            .values
        )

        # Set regularization factors.
        if self.prior_LFC_var is not None:
            ridge_factor = np.diag(1 / self.prior_LFC_var**2)
        else:
            ridge_factor = np.diag(np.repeat(1e-6, num_vars))

        design_matrix = self.design_matrix.values
        LFCs = self.LFC.values

        if not self.quiet:
            print("Running Wald tests...", file=sys.stderr)
        start = time.time()
        pvals, stats, se = self.inference.wald_test(
            design_matrix=design_matrix,
            disp=self.dds.var["dispersions"].values,
            lfc=LFCs,
            mu=mu,
            ridge_factor=ridge_factor,
            contrast=self.contrast_vector,
            lfc_null=np.log(2) * self.lfc_null,  # Convert log2 to natural log
            alt_hypothesis=self.alt_hypothesis,
        )
        end = time.time()
        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.p_values: pd.Series = pd.Series(pvals, index=self.dds.var_names)
        self.statistics: pd.Series = pd.Series(stats, index=self.dds.var_names)
        self.SE: pd.Series = pd.Series(se, index=self.dds.var_names)

        # Account for possible all_zeroes due to outlier refitting in DESeqDataSet
        if self.dds.refit_cooks and self.dds.var["replaced"].sum() > 0:
            self.SE.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.statistics.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.p_values.loc[self.dds.new_all_zeroes_genes] = 1.0

    def run_likelihood_ratio_test(self) -> None:
        r"""Perform a likelihood ratio test (LRT).

        Compares the full model (the design of the ``DeseqDataSet``) to a nested
        ``reduced`` model, using the same gene-wise dispersions for both. This is the
        Python equivalent of R DESeq2's ``DESeq(dds, test="LRT", reduced=...)``.

        For each gene, the test statistic is

        .. math::
            \Lambda = 2 \left( \ell_{\text{full}} - \ell_{\text{reduced}} \right),

        where :math:`\ell` is the negative-binomial log-likelihood evaluated at the
        maximum-likelihood fit of each model. Under the null hypothesis that the extra
        full-model coefficients are zero, :math:`\Lambda` follows a chi-squared
        distribution with degrees of freedom equal to the difference in the number of
        coefficients between the two models. The reported ``log2FoldChange`` and
        ``lfcSE`` still correspond to the requested ``contrast`` of the full model,
        exactly as in R DESeq2.
        """
        if self.shrunk_LFCs and not self.quiet:
            print(
                "Note: running the likelihood ratio test on shrunk LFCs. The LRT is "
                "usually run on the maximum-likelihood (unshrunk) estimates.",
                file=sys.stderr,
            )

        # A reduced design matrix is always set when test="LRT".
        assert self.reduced_design_matrix is not None

        non_zero_idx = self.dds.non_zero_idx
        non_zero_genes = self.dds.non_zero_genes

        full_design_matrix = self.design_matrix.values
        reduced_design_matrix = self.reduced_design_matrix.values
        df = full_design_matrix.shape[1] - reduced_design_matrix.shape[1]

        size_factors = self.dds.obs["size_factors"].values
        disp = self.dds.var["dispersions"].values[non_zero_idx]

        # Counts the full model was fit on: original counts, with Cooks outliers
        # replaced by imputed values for refitted genes (as in R's
        # ``counts(dds, replaced=TRUE)``). This keeps the reduced-model fit and the
        # deviances consistent with the (possibly refitted) full-model LFCs.
        # ``np.array`` (not ``asarray``) to guarantee a copy, so the in-place
        # replacement below never mutates ``dds.X``.
        counts = np.array(self.dds.X, dtype=float)
        if self.dds.refit_cooks and hasattr(self.dds, "counts_to_refit"):
            refit_pos = self.dds.var_names.get_indexer(
                self.dds.counts_to_refit.var_names
            )
            counts[:, refit_pos] = np.asarray(self.dds.counts_to_refit.X, dtype=float)
        counts = counts[:, non_zero_idx]

        # Full-model mean, recomputed from the stored LFCs (natural log scale),
        # matching the convention used by the Wald test.
        lfc = self.LFC.values[non_zero_idx]
        mu_full = np.exp(full_design_matrix @ lfc.T) * size_factors[:, None]

        # Set regularization factors (identical to the Wald test).
        if self.prior_LFC_var is not None:
            ridge_factor = np.diag(1 / self.prior_LFC_var**2)
        else:
            ridge_factor = np.diag(np.repeat(1e-6, full_design_matrix.shape[1]))

        if not self.quiet:
            print("Running LRT tests...", file=sys.stderr)
        start = time.time()

        # Fit the reduced model with the SAME dispersions as the full model.
        _, mu_reduced, _, _ = self.inference.irls(
            counts=counts,
            size_factors=size_factors,
            design_matrix=reduced_design_matrix,
            disp=disp,
            min_mu=self.dds.min_mu,
            beta_tol=self.dds.beta_tol,
        )

        # Standard error of the requested contrast, from the full-model fit, so that
        # the ``lfcSE`` column matches the Wald test output.
        _, _, se = self.inference.wald_test(
            design_matrix=full_design_matrix,
            disp=disp,
            lfc=lfc,
            mu=mu_full,
            ridge_factor=ridge_factor,
            contrast=self.contrast_vector,
            lfc_null=0.0,
            alt_hypothesis=None,
        )

        # LRT statistic and chi-squared p-value. ``nb_nll`` is the negative
        # log-likelihood, so 2 * (ll_full - ll_reduced) = 2 * (nll_reduced - nll_full).
        stats = 2.0 * (nb_nll(counts, mu_reduced, disp) - nb_nll(counts, mu_full, disp))
        # Clip tiny negative values that can arise from numerical noise when the two
        # models fit essentially identically (the true statistic is non-negative).
        stats = np.maximum(stats, 0.0)
        pvals = chi2.sf(stats, df)

        end = time.time()
        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.p_values = pd.Series(np.nan, index=self.dds.var_names)
        self.statistics = pd.Series(np.nan, index=self.dds.var_names)
        self.SE = pd.Series(np.nan, index=self.dds.var_names)
        self.p_values.loc[non_zero_genes] = pvals
        self.statistics.loc[non_zero_genes] = stats
        self.SE.loc[non_zero_genes] = se

        # Account for possible all_zeroes due to outlier refitting in DESeqDataSet
        if self.dds.refit_cooks and self.dds.var["replaced"].sum() > 0:
            self.SE.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.statistics.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.p_values.loc[self.dds.new_all_zeroes_genes] = 1.0

    # TODO update this to reflect the new contrast format
    def lfc_shrink(self, coeff: str, adapt: bool = True) -> None:
        """LFC shrinkage with an apeGLM prior :cite:p:`DeseqStats-zhu2019heavy`.

        Shrinks LFCs using a heavy-tailed Cauchy prior, leaving p-values unchanged.

        Parameters
        ----------
        coeff : str
            The LFC coefficient to shrink. Must be one of the columns of the LFC matrix.
            (default: ``None``).

        adapt: bool
            Whether to use the MLE estimates of LFC to adapt the prior. If False, the
            prior scale is set to 1. (``default=True``)
        """
        if coeff not in self.LFC.columns:
            raise KeyError(
                f"The coeff argument '{coeff}' should be one the LFC columns. "
                f"The available LFC coeffs are {self.LFC.columns[1:]}."
            )

        coeff_idx = self.LFC.columns.get_loc(coeff)

        design_matrix = self.design_matrix.values
        size = 1.0 / self.dds.var["dispersions"].values
        offset = np.log(self.dds.obs["size_factors"]).values

        # Set priors
        prior_no_shrink_scale = 15
        prior_scale = 1
        if adapt:
            prior_var = self._fit_prior_var(coeff_idx=coeff_idx)
            prior_scale = np.minimum(np.sqrt(prior_var), 1)

        if not self.quiet:
            print("Fitting MAP LFCs...", file=sys.stderr)
        start = time.time()
        lfcs, inv_hessians, l_bfgs_b_converged_ = self.inference.lfc_shrink_nbinom_glm(
            design_matrix=design_matrix,
            counts=self.dds.X[:, self.dds.non_zero_idx],
            size=size[self.dds.non_zero_idx],
            offset=offset,
            prior_no_shrink_scale=prior_no_shrink_scale,
            prior_scale=prior_scale,
            optimizer="L-BFGS-B",
            shrink_index=coeff_idx,
        )
        end = time.time()
        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        new_lfc_values = np.array(lfcs)[:, coeff_idx]
        new_se_values = np.array(
            [
                np.sqrt(np.abs(inv_hess[coeff_idx, coeff_idx]))
                for inv_hess in inv_hessians
            ]
        )
        nan_mask = ~np.isfinite(new_lfc_values) | ~np.isfinite(new_se_values)

        if nan_mask.any():
            warnings.warn(
                f"{nan_mask.sum()} gene(s) had NaN/infinite values during LFC shrinkage,"
                " their LFCs and SEs were not updated.",
                UserWarning,
                stacklevel=2,
            )

        # Only update genes with valid (non-NaN) shrinkage results
        valid_genes = self.dds.non_zero_genes[~nan_mask]
        self.LFC.loc[valid_genes, coeff] = new_lfc_values[~nan_mask]
        self.SE.loc[valid_genes] = new_se_values[~nan_mask]

        self._LFC_shrink_converged = pd.Series(
            pd.array([pd.NA] * len(self.dds.var_names), dtype="boolean"),
            index=self.dds.var_names,
        )
        self._LFC_shrink_converged.loc[self.dds.non_zero_genes] = l_bfgs_b_converged_

        # Set a flag to indicate that LFCs were shrunk
        self.shrunk_LFCs = True

        # Replace in results dataframe, if it exists
        if hasattr(self, "results_df"):
            self.results_df["log2FoldChange"] = self.LFC.iloc[:, coeff_idx] / np.log(2)
            self.results_df["lfcSE"] = self.SE / np.log(2)
            if not self.quiet:
                print(f"Shrunk log2 fold change & Wald test p-value: {coeff}")
                print(self.results_df)

    def plot_MA(self, log: bool = True, save_path: str | None = None, **kwargs):
        """
        Create an log ratio (M)-average (A) plot using matplotlib.

        Useful for looking at log fold-change versus mean expression
        between two groups/samples/etc.
        Uses matplotlib to emulate the ``make_MA()`` function in DESeq2 in R.

        Parameters
        ----------
        log : bool
            Whether or not to log scale x and y axes (``default=True``).

        save_path : str, optional
            The path where to save the plot. If left None, the plot won't be saved
            (``default=None``).

        **kwargs
            Matplotlib keyword arguments for the scatter plot.
        """
        # Raise an error if results_df are missing
        if not hasattr(self, "results_df"):
            raise AttributeError(
                "Trying to make an MA plot but p-values were not computed yet. "
                "Please run the summary() method first."
            )

        make_MA_plot(
            self.results_df,
            padj_thresh=self.alpha,
            log=log,
            save_path=save_path,
            lfc_null=self.lfc_null,
            alt_hypothesis=self.alt_hypothesis,
            **kwargs,
        )

    def _independent_filtering(self) -> None:
        """Compute adjusted p-values using independent filtering.

        Corrects p-value trend (see :cite:p:`DeseqStats-love2014moderated`)
        """
        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self._run_test()

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
                result.loc[use, i] = false_discovery_control(U2, method="bh")
        num_rej = (result < self.alpha).sum(axis=0).to_numpy().astype(int)
        lowess_res = lowess(theta, num_rej, frac=1 / 5)

        if num_rej.max() <= 10:
            j = 0
        else:
            residual = num_rej[num_rej > 0] - lowess_res[num_rej > 0]
            thresh = lowess_res.max() - np.sqrt(np.mean(residual**2))
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
            # Estimate p-values with the configured test
            self._run_test()

        self.padj = pd.Series(np.nan, index=self.dds.var_names)
        self.padj.loc[~self.p_values.isna()] = false_discovery_control(
            self.p_values.dropna(), method="bh"
        )

    def _cooks_filtering(self) -> None:
        """Filter p-values based on Cooks outliers."""
        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self._run_test()

        self.p_values[self.dds.cooks_outlier()] = np.nan

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
            return root_scalar(objective, bracket=(min_var, max_var)).root

    def _build_contrast_vector(self) -> None:
        """
        Build a vector corresponding to the desired contrast.

        Allows to test any pair of levels without refitting LFCs.
        """
        factor = self.contrast[0]
        alternative = self.contrast[1]
        ref = self.contrast[2]
        self.contrast_vector = self.dds.contrast(
            column=factor, baseline=ref, group_to_compare=alternative
        )

    def _run_test(self) -> None:
        """Run the configured statistical test (Wald or LRT)."""
        if self.test == "Wald":
            self.run_wald_test()
        else:
            self.run_likelihood_ratio_test()

    def _build_reduced_design_matrix(
        self, reduced: str | np.ndarray | pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Validate and build the reduced-model design matrix for the LRT.

        Parameters
        ----------
        reduced : str, ndarray, pandas.DataFrame, optional
            The reduced model, as passed to the constructor.

        Returns
        -------
        pandas.DataFrame or None
            The reduced-model design matrix (``None`` for the Wald test).
        """
        if self.test == "Wald":
            if reduced is not None:
                raise ValueError(
                    "`reduced` is only used for the likelihood ratio test "
                    "(`test='LRT'`); leave it to None for the Wald test."
                )
            return None

        # test == "LRT"
        if reduced is None:
            raise ValueError("A `reduced` model must be provided when `test='LRT'`.")

        if isinstance(reduced, str):
            if not isinstance(self.dds.design, str):
                raise ValueError(
                    "A formula string can only be used for `reduced` when the "
                    "DeseqDataSet design is itself a formula. Provide `reduced` as a "
                    "design matrix (numpy array or pandas DataFrame) instead."
                )
            reduced_design_matrix = FormulaicContrasts(
                self.dds.obs, reduced
            ).design_matrix
            # The reduced model must be nested in the full model.
            full_columns = set(self.design_matrix.columns)
            if not set(reduced_design_matrix.columns).issubset(full_columns):
                raise ValueError(
                    "The reduced model must be nested in the full model: its columns "
                    f"{list(reduced_design_matrix.columns)} must be a subset of the "
                    f"full design matrix columns {list(self.design_matrix.columns)}."
                )
        elif isinstance(reduced, pd.DataFrame):
            reduced_design_matrix = reduced.copy()
        elif isinstance(reduced, np.ndarray):
            if reduced.ndim != 2 or reduced.shape[0] != self.dds.n_obs:
                raise ValueError(
                    "The reduced design matrix must be 2D with one row per sample "
                    f"({self.dds.n_obs}); got shape {reduced.shape}."
                )
            reduced_design_matrix = pd.DataFrame(
                reduced,
                index=self.design_matrix.index,
                columns=[f"reduced_{i}" for i in range(reduced.shape[1])],
            )
        else:
            raise TypeError(
                "`reduced` must be a formula string, a numpy array, or a pandas "
                f"DataFrame; got {type(reduced).__name__}."
            )

        # The reduced model must have strictly fewer coefficients than the full model.
        n_df = self.design_matrix.shape[1] - reduced_design_matrix.shape[1]
        if n_df < 1:
            raise ValueError(
                "The reduced model must have fewer coefficients than the full model "
                f"(full: {self.design_matrix.shape[1]}, "
                f"reduced: {reduced_design_matrix.shape[1]}). "
                "The likelihood ratio test compares nested models."
            )
        if reduced_design_matrix.shape[0] != self.design_matrix.shape[0]:
            raise ValueError(
                "The reduced design matrix must have one row per sample "
                f"({self.design_matrix.shape[0]}); got "
                f"{reduced_design_matrix.shape[0]}."
            )

        return reduced_design_matrix
