from __future__ import annotations

import sys
import time
import warnings
from typing import Literal
from typing import cast

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
from pandas import Categorical
from pandas import to_numeric
from scipy.optimize import minimize
from scipy.special import polygamma  # type: ignore
from scipy.stats import f  # type: ignore
from scipy.stats import trim_mean  # type: ignore

from pydeseq2.default_inference import DefaultInference
from pydeseq2.inference import Inference
from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform
from pydeseq2.utils import build_design_matrix
from pydeseq2.utils import dispersion_trend
from pydeseq2.utils import make_scatter
from pydeseq2.utils import mean_absolute_deviation
from pydeseq2.utils import n_or_more_replicates
from pydeseq2.utils import nb_nll
from pydeseq2.utils import process_design_factors
from pydeseq2.utils import replace_underscores
from pydeseq2.utils import robust_method_of_moments_disp
from pydeseq2.utils import test_valid_counts
from pydeseq2.utils import trimmed_mean

# Ignore AnnData's FutureWarning about implicit data conversion.
warnings.simplefilter("ignore", FutureWarning)


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
    adata : anndata.AnnData
        AnnData from which to initialize the DeseqDataSet. Must have counts ('X') and
        sample metadata ('obs') fields. If ``None``, both ``counts`` and ``metadata``
        arguments must be provided.

    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.

    metadata : pandas.DataFrame
        DataFrame containing sample metadata.
        Must be indexed by sample barcodes.

    design_factors : str or list
        Name of the columns of metadata to be used as design variables.
        Interaction terms can also be used such as col1:...:colN.
        Finally, if you wish to use a formula, you can write:
        ``"~ col1 + ... + colN + colk1:...:colN1 + ... + colkni:...:colNni"``.
        In this case, the formula will be parsed and the factors (interacting
        and non-interacting) will be extracted automatically from adata or counts
        and metadata relying on the wonderful formulaic package. Note that for
        the moment pydeseq2 doesn't support the entirety of formulaic grammar.
        Formulas using syntax such as 'C(values, contr.poly, levels=[10, 20, 30])'
        or '*' are NOT supported yet as input. For precise control over categorical
        encoding users should use ref_level argument. Every column will be interpreted
        as categorical by design except if listed in continuous_factors.
        (default: ``"condition"``).

    continuous_factors : list or None
        An optional list of continuous (as opposed to categorical) factors. Any factor
        not in ``continuous_factors`` will be considered categorical (default: ``None``).

    design_matrix: pd.DataFrame or None
        If given will take precedence over design_factors and ref_level. The columns
        must be one-hot encoded and follow the "covariate_test_vs_ref" format. The
        index must be the sample names. (default: ``None``).

    ref_level : list or None
        An optional list of tuples each with two strings of the form
        ``("factor", "test_level")`` specifying the factor of interest and the
        reference (control) level against which we're testing, e.g.
        ``[("condition", "A")]``. (default: ``None``).

    fit_type: str
        Either ``"parametric"`` or ``"mean"`` for the type of fitting of dispersions to
        the mean intensity. ``"parametric"``: fit a dispersion-mean relation via a
        robust gamma-family GLM. ``"mean"``: use the mean of gene-wise dispersion
        estimates. Will set the fit type for the DEA and the vst transformation. If
        needed, it can be set separately for each method.(default: ``"parametric"``).

    min_mu : float
        Threshold for mean estimates. (default: ``0.5``).

    min_disp : float
        Lower threshold for dispersion parameters. (default: ``1e-8``).

    max_disp : float
        Upper threshold for dispersion parameters.
        Note: The threshold that is actually enforced is max(max_disp, len(counts)).
        (default: ``10``).

    refit_cooks : bool
        Whether to refit cooks outliers. (default: ``True``).

    min_replicates : int
        Minimum number of replicates a condition should have
        to allow refitting its samples. (default: ``7``).

    beta_tol : float
        Stopping criterion for IRWLS. (default: ``1e-8``).

        .. math:: \vert dev_t - dev_{t+1}\vert / (\vert dev \vert + 0.1) < \beta_{tol}.

    n_cpus : int
        Number of cpus to use.  If ``None`` and if ``inference`` is not provided, all
        available cpus will be used by the ``DefaultInference``. If both are specified
        (i.e., ``n_cpus`` and ``inference`` are not ``None``), it will try to override
        the ``n_cpus`` attribute of the ``inference`` object. (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).

    quiet : bool
        Suppress deseq2 status updates during fit.

    low_memory : bool
        Remove intermediate data structures from .layers and from .obsm that are no
        longer necessary after they are used during deseq2 run, such as Cook's
        distances. (default: False)

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
        Key-indexed multi-dimensional gene annotation of length ‘number of genes’.
        Stores "dispersions" and "LFC", among others.

    layers
        Key-indexed multi-dimensional arrays aligned to dimensions of `X`, e.g. "cooks".

    n_processes : int
        Number of cpus to use for multiprocessing.

    non_zero_idx : ndarray
        Indices of genes that have non-uniformly zero counts.

    non_zero_genes : pandas.Index
        Index of genes that have non-uniformly zero counts.

    counts_to_refit : anndata.AnnData
        Read counts after replacement, containing only genes
        for which dispersions and LFCs must be fitted again.

    new_all_zeroes_genes : pandas.Index
        Genes which have only zero counts after outlier replacement.

    quiet : bool
        Suppress deseq2 status updates during fit.

    logmeans: numpy.ndarray
        Gene-wise mean log counts, computed in ``preprocessing.deseq2_norm_fit()``.

    filtered_genes: numpy.ndarray
        Genes whose log means are different from -∞, computed in
        preprocessing.deseq2_norm_fit().

    References
    ----------
    .. bibliography::
        :keyprefix: DeseqDataSet-

    """

    def __init__(
        self,
        *,
        adata: ad.AnnData | None = None,
        counts: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        design_factors: str | list[str] = "condition",
        continuous_factors: list[str] | None = None,
        design_matrix: pd.DataFrame | None = None,
        ref_level: list[tuple[str, str]] | None = None,
        fit_type: Literal["parametric", "mean"] = "parametric",
        min_mu: float = 0.5,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        refit_cooks: bool = True,
        min_replicates: int = 7,
        beta_tol: float = 1e-8,
        n_cpus: int | None = None,
        inference: Inference | None = None,
        quiet: bool = False,
        low_memory: bool = False,
    ) -> None:
        # Initialize the AnnData part
        if adata is not None:
            if counts is not None:
                warnings.warn(
                    "adata was provided; ignoring counts.", UserWarning, stacklevel=2
                )
            if metadata is not None:
                warnings.warn(
                    "adata was provided; ignoring metadata.", UserWarning, stacklevel=2
                )
            # Test counts before going further
            test_valid_counts(adata.X)
            # Copy fields from original AnnData
            self.__dict__.update(adata.__dict__)
            # Cast counts to ints to avoid any issue
            self.X = adata.X.astype(int)
        elif counts is not None and metadata is not None:
            # Test counts before going further
            test_valid_counts(counts)
            super().__init__(X=counts.astype(int), obs=metadata)
        else:
            raise ValueError(
                "Either adata or both counts and metadata arguments must be provided."
            )
        for col in self.obs.columns:
            if col.strip() != col:
                raise ValueError("Columns cannot contain leading or trailing spaces")

        self.fit_type = fit_type

        continuous_factors = continuous_factors if continuous_factors is not None else []
        if type(design_factors) is list:
            warnings.warn(
                """Starting from version 0.5.0, PyDESeq2 will no longer accept lists as
                design factors, only strings representing wilkinson formulae
                (e.g., '~ condition + treatment + condition:treatment') .""",
                DeprecationWarning,
                stacklevel=2,
            )

        (
            self.design_formula,
            self.design_factors_list,
            self.single_design_factors,
            self.continuous_factors,
            self.ref_level,
        ) = process_design_factors(
            design_factors,
            self.obs.columns,
            continuous_factors,
            ref_level,
            design_matrix,
        )

        # Replace underscores in column names with hyphens
        replace_dict = {
            col: col.replace("_", "-")
            for col in self.obs.columns
            if col.replace("_", "-") in self.single_design_factors
        }
        self.obs.rename(
            columns=replace_dict,
            inplace=True,
        )

        for factor in self.single_design_factors:
            if self.obs[factor].isna().any():
                raise ValueError("NaNs are not allowed in the design factors.")
            if factor not in self.continuous_factors:
                self.obs[factor] = Categorical(self.obs[factor].astype(str))
                levels = self.obs[factor].unique()
                if "_" in "".join(levels):
                    warnings.warn(
                        f"""Some categories in the design of factor {factor} contain
                            underscores ('_'). They will be converted to hyphens
                            ('-').""",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.obs[factor] = self.obs[factor].cat.rename_categories(
                        replace_underscores(levels.to_list())
                    )

            else:
                self.obs[factor] = to_numeric(self.obs[factor])

        if design_matrix is not None:
            self.obsm["design_matrix"] = design_matrix
        else:
            self.obsm["design_matrix"] = build_design_matrix(
                metadata=self.obs,
                design_factors=self.design_formula,
                ref_level=self.ref_level,
            )

        if self.obs[self.single_design_factors].isna().any().any():
            raise ValueError("NaNs are not allowed in the design factors.")

        if len(self.single_design_factors) == 1:
            if self.obs[self.single_design_factors[0]].nunique() == 1:
                raise ValueError("Only one unique value and one design factor.")

        # Check that the design matrix has full rank
        self._check_full_rank_design()

        self.min_mu = min_mu
        self.min_disp = min_disp
        self.max_disp = np.maximum(max_disp, self.n_obs)
        self.refit_cooks = refit_cooks
        self.ref_level = ref_level
        self.min_replicates = min_replicates
        self.beta_tol = beta_tol
        self.quiet = quiet
        self.low_memory = low_memory
        self.logmeans = None
        self.filtered_genes = None

        if inference:
            if hasattr(inference, "n_cpus"):
                if n_cpus:
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

    def vst(
        self,
        use_design: bool = False,
        fit_type: Literal["parametric", "mean"] | None = None,
    ) -> None:
        """Fit a variance stabilizing transformation, and apply it to normalized counts.

        Results are stored in ``dds.layers["vst_counts"]``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used. (default: ``False``).

        fit_type: str
            * ``None``: fit_type provided at initialization to fit
              the dispersions trend curve.
            * ``"parametric"``: fit a dispersion-mean relation via a robust
              gamma-family GLM.
            * ``"mean"``: use the mean of gene-wise dispersion estimates.

            (default: ``None``).
        """
        if fit_type is not None:
            self.vst_fit_type = fit_type
        else:
            self.vst_fit_type = self.fit_type

        print(f"Fit type used for VST : {self.vst_fit_type}")

        self.vst_fit(use_design=use_design)
        self.layers["vst_counts"] = self.vst_transform()

    def vst_fit(
        self,
        use_design: bool = False,
    ) -> None:
        """Fit a variance stabilizing transformation.

        This method should be called before `vst_transform`.

        Results are stored in ``dds.layers["vst_counts"]``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used.
            Only useful if ``fit_type = "parametric"`.
            (default: ``False``).
        """
        # Start by fitting median-of-ratio size factors if not already present,
        # or if they were computed iteratively
        if "size_factors" not in self.obsm or self.logmeans is None:
            self.fit_size_factors()  # by default, fit_type != "iterative"

        if not hasattr(self, "vst_fit_type"):
            self.vst_fit_type = self.fit_type

        if use_design:
            if self.vst_fit_type == "parametric":
                self._fit_parametric_dispersion_trend(vst=True)
            else:
                warnings.warn(
                    "use_design=True is only useful when fit_type='parametric'. ",
                    UserWarning,
                    stacklevel=2,
                )
                self.fit_genewise_dispersions(vst=True)

        else:
            # Reduce the design matrix to an intercept and reconstruct at the end
            self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
            self.obsm["design_matrix"] = pd.DataFrame(
                1, index=self.obs_names, columns=["intercept"]
            )
            # Fit the trend curve with an intercept design
            self.fit_genewise_dispersions(vst=True)
            if self.vst_fit_type == "parametric":
                self._fit_parametric_dispersion_trend(vst=True)

            # Restore the design matrix and free buffer
            self.obsm["design_matrix"] = self.obsm["design_matrix_buffer"].copy()
            del self.obsm["design_matrix_buffer"]

    def vst_transform(self, counts: np.ndarray | None = None) -> np.ndarray:
        """Apply the variance stabilizing transformation.

        Uses the results from the ``vst_fit`` method.

        Parameters
        ----------
        counts : numpy.ndarray
            Counts to transform. If ``None``, use the counts from the current dataset.
            (default: ``None``).

        Returns
        -------
        numpy.ndarray
            Variance stabilized counts.

        Raises
        ------
        RuntimeError
            If the size factors were not fitted before calling this method.
        """
        if "size_factors" not in self.obsm:
            raise RuntimeError(
                "The vst_fit method should be called prior to vst_transform."
            )

        if counts is None:
            # the transformed counts will be the current ones
            normed_counts = self.layers["normed_counts"]
        else:
            if self.logmeans is None:
                # the size factors were still computed iteratively
                warnings.warn(
                    "The size factors were fitted iteratively. They will "
                    "be re-computed with the counts to be transformed. In a train/test "
                    "setting with a downstream task, this would result in a leak of "
                    "data from test to train set.",
                    UserWarning,
                    stacklevel=2,
                )
                logmeans, filtered_genes = deseq2_norm_fit(counts)
            else:
                logmeans, filtered_genes = self.logmeans, self.filtered_genes

            normed_counts, _ = deseq2_norm_transform(counts, logmeans, filtered_genes)

        if self.vst_fit_type == "parametric":
            if "vst_trend_coeffs" not in self.uns:
                raise RuntimeError("Fit the dispersion curve prior to applying VST.")

            a0, a1 = self.uns["vst_trend_coeffs"]
            return np.log2(
                (
                    1
                    + a1
                    + 2 * a0 * normed_counts
                    + 2 * np.sqrt(a0 * normed_counts * (1 + a1 + a0 * normed_counts))
                )
                / (4 * a0)
            )
        elif self.vst_fit_type == "mean":
            gene_dispersions = self.varm["vst_genewise_dispersions"]
            use_for_mean = gene_dispersions > 10 * self.min_disp
            mean_disp = trim_mean(gene_dispersions[use_for_mean], proportiontocut=0.001)
            return (
                2 * np.arcsinh(np.sqrt(mean_disp * normed_counts))
                - np.log(mean_disp)
                - np.log(4)
            ) / np.log(2)
        else:
            raise NotImplementedError(
                f"Found fit_type '{self.vst_fit_type}'. Expected 'parametric' or 'mean'."
            )

    def deseq2(self, fit_type: Literal["parametric", "mean"] | None = None) -> None:
        """Perform dispersion and log fold-change (LFC) estimation.

        Wrapper for the first part of the PyDESeq2 pipeline.


        Parameters
        ----------
        fit_type : str
            Either None, ``"parametric"`` or ``"mean"`` for the type of fitting of
            dispersions to the mean intensity.``"parametric"``: fit a dispersion-mean
            relation via a robust gamma-family GLM. ``"mean"``: use the mean of
            gene-wise dispersion estimates.

            If None, the fit_type provided at class initialization is used.
            (default: ``None``).
        """
        if fit_type is not None:
            self.fit_type = fit_type
            print(f"Using {self.fit_type} fit type.")

        # Compute DESeq2 normalization factors using the Median-of-ratios method
        self.fit_size_factors()
        # Fit an independent negative binomial model per gene
        self.fit_genewise_dispersions()
        # Fit a parameterized trend curve for dispersions, of the form
        # f(\mu) = \alpha_1/\mu + a_0
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

        # Compute gene mask for cooks outliers
        self.cooks_outlier()

    def fit_size_factors(
        self,
        fit_type: Literal["ratio", "poscounts", "iterative"] = "ratio",
        control_genes: np.ndarray | list[str] | list[int] | pd.Index | None = None,
    ) -> None:
        """Fit sample-wise deseq2 normalization (size) factors.

        Uses the median-of-ratios method: see :func:`pydeseq2.preprocessing.deseq2_norm`,
        unless each gene has at least one sample with zero read counts, in which case it
        switches to the ``iterative`` method.

        Also available is the 'poscounts' method implemented in DESeq2 for the
        single-cell or metagenomics use case where there may be few or no features which
        have no zero values. In this situation, size factors can depend on a very small
        number of features (or only one feature) leading to incorrect inference. This
        method for calculating size factors will only exclude genes which have all-0
        values (and are not amenable to inference anyway).

        The "poscounts" method calculates the n-th root of the product of the non-zero
        (positive) counts.

        Control genes can be optionally provided; if so, size factors will be fit to
        only the genes in this argument. This is the same functionality as controlGenes
        in R DESeq2. Any valid AnnData indexer (bool, int position, var_name string) is
        accepted.

        Parameters
        ----------
        fit_type : str
            The normalization method to use: "ratio", "poscounts" or "iterative".
            (default: ``"ratio"``).
        control_genes : ndarray, list, pandas.Index, or None
            Genes to use as control genes for size factor fitting. If None, all genes
            are used. (default: ``None``).
        """
        if not self.quiet:
            print("Fitting size factors...", file=sys.stderr)

        start = time.time()

        # If control genes are provided, set a mask where those genes are True
        if control_genes is not None:
            _control_mask = np.zeros(self.X.shape[1], dtype=bool)

            # Use AnnData internal indexing to get gene index array
            # Allows bool/int/var_name to be provided
            _control_mask[self._normalize_indices((slice(None), control_genes))[1]] = (
                True
            )

        # Otherwise mask all genes to be True
        else:
            _control_mask = np.ones(self.X.shape[1], dtype=bool)

        if fit_type == "iterative":
            self._fit_iterate_size_factors()

        elif fit_type == "poscounts":
            # Calculate logcounts for x > 0 and take the mean for each gene
            log_counts = np.zeros_like(self.X, dtype=float)
            np.log(self.X, out=log_counts, where=self.X != 0)
            logmeans = log_counts.mean(0)

            # Determine which genes are usable (finite logmeans)
            self.filtered_genes = (~np.isinf(logmeans)) & (logmeans > 0)
            _control_mask &= self.filtered_genes

            # Calculate size factor per sample
            def sizeFactor(x):
                _mask = np.logical_and(_control_mask, x > 0)
                return np.exp(np.median(np.log(x[_mask]) - logmeans[_mask]))

            sf = np.apply_along_axis(sizeFactor, 1, self.X)
            del log_counts

            # Normalize size factors to a geometric mean of 1 to match DESeq
            self.obsm["size_factors"] = sf / (np.exp(np.mean(np.log(sf))))
            self.layers["normed_counts"] = self.X / self.obsm["size_factors"][:, None]
            self.logmeans = logmeans

        # Test whether it is possible to use median-of-ratios.
        elif (self.X == 0).any(0).all():
            # There is at least a zero for each gene
            warnings.warn(
                "Every gene contains at least one zero, "
                "cannot compute log geometric means. Switching to iterative mode.",
                UserWarning,
                stacklevel=2,
            )
            self._fit_iterate_size_factors()

        else:
            self.logmeans, self.filtered_genes = deseq2_norm_fit(self.X)
            _control_mask &= self.filtered_genes

            (
                self.layers["normed_counts"],
                self.obsm["size_factors"],
            ) = deseq2_norm_transform(self.X, self.logmeans, _control_mask)

        end = time.time()
        self.varm["_normed_means"] = self.layers["normed_counts"].mean(0)

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

    def fit_genewise_dispersions(self, vst=False) -> None:
        """Fit gene-wise dispersion estimates.

        Fits a negative binomial per gene, independently.

        Parameters
        ----------
        vst : bool
            Whether the dispersion estimates are being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        # Check that size factors are available. If not, compute them.
        if "size_factors" not in self.obsm:
            self.fit_size_factors()

        # Exclude genes with all zeroes
        self.varm["non_zero"] = ~(self.X == 0).all(axis=0)
        self.non_zero_idx = np.arange(self.n_vars)[self.varm["non_zero"]]
        self.non_zero_genes = self.var_names[self.varm["non_zero"]]

        if isinstance(self.non_zero_genes, pd.MultiIndex):
            raise ValueError("non_zero_genes should not be a MultiIndex")

        # Fit "method of moments" dispersion estimates
        self._fit_MoM_dispersions()

        # Convert design_matrix to numpy for speed
        design_matrix = self.obsm["design_matrix"].values

        # mu_hat is initialized differently depending on the number of different factor
        # groups. If there are as many different factor combinations as design factors
        # (intercept included), it is fitted with a linear model, otherwise it is fitted
        # with a GLM (using rough dispersion estimates).
        if (
            len(self.obsm["design_matrix"].value_counts())
            == self.obsm["design_matrix"].shape[-1]
        ):
            mu_hat_ = self.inference.lin_reg_mu(
                counts=self.X[:, self.non_zero_idx],
                size_factors=self.obsm["size_factors"],
                design_matrix=design_matrix,
                min_mu=self.min_mu,
            )
        else:
            _, mu_hat_, _, _ = self.inference.irls(
                counts=self.X[:, self.non_zero_idx],
                size_factors=self.obsm["size_factors"],
                design_matrix=design_matrix,
                disp=self.varm["_MoM_dispersions"][self.non_zero_idx],
                min_mu=self.min_mu,
                beta_tol=self.beta_tol,
            )

        mu_param_name = "_vst_mu_hat" if vst else "_mu_hat"
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        self.layers[mu_param_name] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers[mu_param_name][:, self.varm["non_zero"]] = mu_hat_

        del mu_hat_

        if not self.quiet:
            print("Fitting dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=self.X[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers[mu_param_name][:, self.non_zero_idx],
            alpha_hat=self.varm["_MoM_dispersions"][self.non_zero_idx],
            min_disp=self.min_disp,
            max_disp=self.max_disp,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.varm[disp_param_name] = np.full(self.n_vars, np.nan)
        self.varm[disp_param_name][self.varm["non_zero"]] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_genewise_converged"] = np.full(self.n_vars, np.nan)
        self.varm["_genewise_converged"][self.varm["non_zero"]] = l_bfgs_b_converged_

    def fit_dispersion_trend(self, vst: bool = False) -> None:
        """Fit the dispersion trend curve.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        fit_type = self.vst_fit_type if vst else self.fit_type

        # Check that genewise dispersions are available. If not, compute them.
        if disp_param_name not in self.varm:
            self.fit_genewise_dispersions(vst)

        if not self.quiet:
            print("Fitting dispersion trend curve...", file=sys.stderr)
        start = time.time()

        if fit_type == "parametric":
            self._fit_parametric_dispersion_trend(vst)
        elif fit_type == "mean":
            self._fit_mean_dispersion_trend(vst)
        else:
            raise NotImplementedError(
                f"Expected 'parametric' or 'mean' trend curve fit "
                f"types, received {fit_type}"
            )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

    def disp_function(self, x):
        """Return the dispersion trend function at x."""
        if self.uns["disp_function_type"] == "parametric":
            return dispersion_trend(x, self.uns["trend_coeffs"])
        elif self.uns["disp_function_type"] == "mean":
            return np.full_like(x, self.uns["mean_disp"])

    def fit_dispersion_prior(self) -> None:
        """Fit dispersion variance priors and standard deviation of log-residuals.

        The computation is based on genes whose dispersions are above 100 * min_disp.

        Note: when the design matrix has fewer than 3 degrees of freedom, the
        estimate of log dispersions is likely to be imprecise.
        """
        # Check that the dispersion trend curve was fitted. If not, fit it.
        if "fitted_dispersions" not in self.varm:
            self.fit_dispersion_trend()

        # Exclude genes with all zeroes
        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[-1]

        # Check the degrees of freedom
        if (num_samples - num_vars) <= 3:
            warnings.warn(
                "As the residual degrees of freedom is less than 3, the distribution "
                "of log dispersions is especially asymmetric and likely to be poorly "
                "estimated by the MAD.",
                UserWarning,
                stacklevel=2,
            )

        # Fit dispersions to the curve, and compute log residuals
        disp_residuals = np.log(
            self[:, self.non_zero_genes].varm["genewise_dispersions"]
        ) - np.log(self[:, self.non_zero_genes].varm["fitted_dispersions"])

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = self[:, self.non_zero_genes].varm["genewise_dispersions"] >= (
            100 * self.min_disp
        )

        self.uns["_squared_logres"] = (
            mean_absolute_deviation(disp_residuals[above_min_disp]) ** 2
        )

        self.uns["prior_disp_var"] = np.maximum(
            self.uns["_squared_logres"] - polygamma(1, (num_samples - num_vars) / 2),
            0.25,
        )

    def fit_MAP_dispersions(self) -> None:
        """Fit Maximum a Posteriori dispersion estimates.

        After MAP dispersions are fit, filter genes for which we don't apply shrinkage.
        """
        # Check that the dispersion prior variance is available. If not, compute it.
        if "prior_disp_var" not in self.uns:
            self.fit_dispersion_prior()

        # Convert design matrix to numpy for speed
        design_matrix = self.obsm["design_matrix"].values

        if not self.quiet:
            print("Fitting MAP dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=self.X[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers["_mu_hat"][:, self.non_zero_idx],
            alpha_hat=self.varm["fitted_dispersions"][self.non_zero_idx],
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            prior_disp_var=self.uns["prior_disp_var"].item(),
            cr_reg=True,
            prior_reg=True,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end-start:.2f} seconds.\n", file=sys.stderr)

        self.varm["MAP_dispersions"] = np.full(self.n_vars, np.nan)
        self.varm["MAP_dispersions"][self.varm["non_zero"]] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_MAP_converged"] = np.full(self.n_vars, np.nan)
        self.varm["_MAP_converged"][self.varm["non_zero"]] = l_bfgs_b_converged_

        # Filter outlier genes for which we won't apply shrinkage
        self.varm["dispersions"] = self.varm["MAP_dispersions"].copy()
        self.varm["_outlier_genes"] = np.log(self.varm["genewise_dispersions"]) > np.log(
            self.varm["fitted_dispersions"]
        ) + 2 * np.sqrt(self.uns["_squared_logres"])
        self.varm["dispersions"][self.varm["_outlier_genes"]] = self.varm[
            "genewise_dispersions"
        ][self.varm["_outlier_genes"]]

        if self.low_memory:
            del self.layers["_mu_hat"]

    def fit_LFC(self) -> None:
        """Fit log fold change (LFC) coefficients.

        In the 2-level setting, the intercept corresponds to the base mean,
        while the second is the actual LFC coefficient, in natural log scale.
        """
        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()

        # Convert design matrix to numpy for speed
        design_matrix = self.obsm["design_matrix"].values

        if not self.quiet:
            print("Fitting LFCs...", file=sys.stderr)
        start = time.time()
        mle_lfcs_, mu_, hat_diagonals_, converged_ = self.inference.irls(
            counts=self.X[:, self.non_zero_idx],
            size_factors=self.obsm["size_factors"],
            design_matrix=design_matrix,
            disp=self.varm["dispersions"][self.non_zero_idx],
            min_mu=self.min_mu,
            beta_tol=self.beta_tol,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end-start:.2f} seconds.\n", file=sys.stderr)

        self.varm["LFC"] = pd.DataFrame(
            np.nan,
            index=self.var_names,
            columns=self.obsm["design_matrix"].columns,
        )

        self.varm["LFC"].update(
            pd.DataFrame(
                mle_lfcs_,
                index=self.non_zero_genes,
                columns=self.obsm["design_matrix"].columns,
            )
        )

        self.obsm["_mu_LFC"] = mu_
        self.obsm["_hat_diagonals"] = hat_diagonals_

        self.varm["_LFC_converged"] = np.full(self.n_vars, np.nan)
        self.varm["_LFC_converged"][self.varm["non_zero"]] = converged_

    def calculate_cooks(self) -> None:
        """Compute Cook's distance for outlier detection.

        Measures the contribution of a single entry to the output of LFC estimation.
        """
        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()

        if not self.quiet:
            print("Calculating cook's distance...", file=sys.stderr)

        start = time.time()
        num_vars = self.obsm["design_matrix"].shape[-1]

        # Calculate dispersion
        dispersions = robust_method_of_moments_disp(
            self.layers["normed_counts"][:, self.varm["non_zero"]],
            self.obsm["design_matrix"],
        )

        # Calculate the squared pearson residuals for non-zero features
        squared_pearson_res = self.X[:, self.varm["non_zero"]] - self.obsm["_mu_LFC"]
        squared_pearson_res **= 2

        # Calculate the overdispersion parameter tau
        V = self.obsm["_mu_LFC"] ** 2
        V *= dispersions[None, :]
        V += self.obsm["_mu_LFC"]

        # Calculate r^2 / (tau * num_vars)
        squared_pearson_res /= V
        squared_pearson_res /= num_vars

        del V

        # Calculate leverage modifier H / (1 - H)^2
        diag_mul = 1 - self.obsm["_hat_diagonals"]
        diag_mul **= 2
        diag_mul = self.obsm["_hat_diagonals"] / diag_mul

        # Multiply r^2 / (tau * num_vars) by H / (1 - H)^2 to get cook's distance
        squared_pearson_res *= diag_mul

        del diag_mul

        if self.low_memory:
            del self.obsm["_mu_LFC"]
            del self.obsm["_hat_diagonals"]

        self.layers["cooks"] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers["cooks"][:, self.varm["non_zero"]] = squared_pearson_res

        if not self.quiet:
            print(f"... done in {time.time()-start:.2f} seconds.\n", file=sys.stderr)

    def refit(self) -> None:
        """Refit Cook outliers.

        Replace values that are filtered out based on the Cooks distance with imputed
        values, and then re-run the whole DESeq2 pipeline on replaced values.
        """
        # Replace outlier counts
        self._replace_outliers()
        if not self.quiet:
            print(
                f"Replacing {sum(self.varm['replaced']) } outlier genes.\n",
                file=sys.stderr,
            )

        if sum(self.varm["replaced"]) > 0:
            # Refit dispersions and LFCs for genes that had outliers replaced
            self._refit_without_outliers()
        else:
            # Store the fact that no sample was refitted
            self.varm["refitted"] = np.full(
                self.n_vars,
                False,
            )

    def cooks_outlier(self):
        """Filter p-values based on Cooks outliers."""
        if "_pvalue_cooks_outlier" in self.varm.keys():
            return self.varm["_pvalue_cooks_outlier"]

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[-1]
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)

        # As in DESeq2, only take samples with 3 or more replicates when looking for
        # max cooks.
        use_for_max = n_or_more_replicates(self.obsm["design_matrix"], 3)

        # If for a gene there are 3 samples or more that have more counts than the
        # maximum cooks sample, don't count this gene as an outlier.

        # Take into account whether we already replaced outliers
        if (
            self.refit_cooks
            and (self.varm["refitted"].sum() > 0)
            and "replace_cooks" in self.layers.keys()
        ):
            cooks_outlier = (
                self.layers["replace_cooks"][use_for_max, :] > cooks_cutoff
            ).any(axis=0)

        else:
            cooks_outlier = (self.layers["cooks"][use_for_max, :] > cooks_cutoff).any(
                axis=0
            )

        pos = self.layers["cooks"][:, cooks_outlier].argmax(0)

        cooks_outlier[cooks_outlier] = (
            self.X[:, cooks_outlier] > self.X[:, cooks_outlier][pos, np.arange(len(pos))]
        ).sum(0) < 3

        if self.low_memory:
            del self.layers["cooks"]

        if self.low_memory and "replace_cooks" in self.layers.keys():
            del self.layers["replace_cooks"]

        self.varm["_pvalue_cooks_outlier"] = cooks_outlier
        return self.varm["_pvalue_cooks_outlier"]

    def _fit_MoM_dispersions(self) -> None:
        """Rough method of moments initial dispersions fit.

        Estimates are the max of "robust" and "method of moments" estimates.
        """
        # Check that size_factors are available. If not, compute them.
        if "normed_counts" not in self.layers:
            self.fit_size_factors()

        normed_counts = self.layers["normed_counts"][:, self.non_zero_idx]
        rde = self.inference.fit_rough_dispersions(
            normed_counts,
            self.obsm["design_matrix"].values,
        )
        mde = self.inference.fit_moments_dispersions(
            normed_counts, self.obsm["size_factors"]
        )
        alpha_hat = np.minimum(rde, mde)

        self.varm["_MoM_dispersions"] = np.full(self.n_vars, np.nan)
        self.varm["_MoM_dispersions"][self.varm["non_zero"]] = np.clip(
            alpha_hat, self.min_disp, self.max_disp
        )

    def plot_dispersions(
        self, log: bool = True, save_path: str | None = None, **kwargs
    ) -> None:
        """Plot dispersions.

        Make a scatter plot with genewise dispersions, trend curve and final (MAP)
        dispersions.

        Parameters
        ----------
        log : bool
            Whether to log scale x and y axes (``default=True``).

        save_path : str or None
            The path where to save the plot. If left None, the plot won't be saved
            (``default=None``).

        **kwargs
            Keyword arguments for the scatter plot.
        """
        disps = [
            self.varm["genewise_dispersions"],
            self.varm["dispersions"],
            self.varm["fitted_dispersions"],
        ]
        legend_labels = ["Estimated", "Final", "Fitted"]
        make_scatter(
            disps,
            legend_labels=legend_labels,
            x_val=self.varm["_normed_means"],
            log=log,
            save_path=save_path,
            **kwargs,
        )

    def _fit_parametric_dispersion_trend(self, vst: bool = False):
        r"""Fit the dispersion curve according to a parametric model.

        :math:`f(\mu) = \alpha_1/\mu + a_0`.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        if disp_param_name not in self.varm:
            self.fit_genewise_dispersions(vst)

        # Exclude all-zero counts
        targets = pd.Series(
            self[:, self.non_zero_genes].varm[disp_param_name].copy(),
            index=self.non_zero_genes,
        )
        covariates = pd.Series(
            1 / self[:, self.non_zero_genes].varm["_normed_means"],
            index=self.non_zero_genes,
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
        while (coeffs > 1e-10).all() and (
            np.log(np.abs(coeffs / old_coeffs)) ** 2
        ).sum() >= 1e-6:
            old_coeffs = coeffs
            coeffs, predictions, converged = self.inference.dispersion_trend_gamma_glm(
                covariates, targets
            )

            if not converged or (coeffs <= 1e-10).any():
                warnings.warn(
                    "The dispersion trend curve fitting did not converge. "
                    "Switching to a mean-based dispersion trend.",
                    UserWarning,
                    stacklevel=2,
                )

                self._fit_mean_dispersion_trend(vst)
                return

            # Filter out genes that are too far away from the curve before refitting
            pred_ratios = self[:, covariates.index].varm[disp_param_name] / predictions

            targets.drop(
                targets[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )
            covariates.drop(
                covariates[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )

        if vst:
            self.uns["vst_trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])
        else:
            self.uns["trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])

            self.varm["fitted_dispersions"] = np.full(self.n_vars, np.nan)
            self.uns["disp_function_type"] = "parametric"
            self.varm["fitted_dispersions"][self.varm["non_zero"]] = self.disp_function(
                self.varm["_normed_means"][self.varm["non_zero"]]
            )

    def _fit_mean_dispersion_trend(self, vst: bool = False):
        """Use the mean of dispersions as trend curve.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        self.uns["mean_disp"] = trim_mean(
            self.varm[disp_param_name][self.varm[disp_param_name] > 10 * self.min_disp],
            proportiontocut=0.001,
        )

        if vst:
            self.vst_fit_type = "mean"
        else:
            self.uns["disp_function_type"] = "mean"
        self.varm["fitted_dispersions"] = np.full(self.n_vars, self.uns["mean_disp"])

    def _replace_outliers(self) -> None:
        """Replace values that are filtered out (based on Cooks) with imputed values."""
        # Check that cooks distances are available. If not, compute them.
        if "cooks" not in self.layers:
            self.calculate_cooks()

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[1]

        # Check whether cohorts have enough samples to allow refitting
        self.obsm["replaceable"] = n_or_more_replicates(
            self.obsm["design_matrix"], self.min_replicates
        ).values

        if self.obsm["replaceable"].sum() == 0:
            # No sample can be replaced. Set self.replaced to False and exit.
            self.varm["replaced"] = np.full(
                self.n_vars,
                False,
            )
            return

        # Get positions of counts with cooks above threshold
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)
        idx = self.layers["cooks"] > cooks_cutoff
        self.varm["replaced"] = idx.any(axis=0)

        if sum(self.varm["replaced"] > 0):
            # Compute replacement counts: trimmed means * size_factors
            self.counts_to_refit = self[:, self.varm["replaced"]].copy()

            trim_base_mean = pd.DataFrame(
                cast(
                    np.ndarray,
                    trimmed_mean(
                        self.counts_to_refit.X / self.obsm["size_factors"][:, None],
                        trim=0.2,
                        axis=0,
                    ),
                ),
                index=self.counts_to_refit.var_names,
            )

            replacement_counts = (
                pd.DataFrame(
                    trim_base_mean.values * self.obsm["size_factors"],
                    index=self.counts_to_refit.var_names,
                    columns=self.counts_to_refit.obs_names,
                )
                .astype(int)
                .T
            )

            self.counts_to_refit.X[
                self.obsm["replaceable"][:, None] & idx[:, self.varm["replaced"]]
            ] = replacement_counts.values[
                self.obsm["replaceable"][:, None] & idx[:, self.varm["replaced"]]
            ]

    def _refit_without_outliers(
        self,
    ) -> None:
        """Re-run the whole DESeq2 pipeline with replaced outliers."""
        assert (
            self.refit_cooks
        ), "Trying to refit Cooks outliers but the 'refit_cooks' flag is set to False"

        # Check that _replace_outliers() was previously run.
        if "replaced" not in self.varm:
            self._replace_outliers()

        # Only refit genes for which replacing outliers hasn't resulted in all zeroes
        new_all_zeroes = (self.counts_to_refit.X == 0).all(axis=0)
        self.new_all_zeroes_genes = self.counts_to_refit.var_names[new_all_zeroes]

        self.varm["refitted"] = self.varm["replaced"].copy()
        # Only replace if genes are not all zeroes after outlier replacement
        self.varm["refitted"][self.varm["refitted"]] = ~new_all_zeroes

        # Take into account new all-zero genes
        if new_all_zeroes.sum() > 0:
            self.varm["_normed_means"][
                self.var_names.get_indexer(self.new_all_zeroes_genes)
            ] = 0
            self.varm["LFC"].loc[self.new_all_zeroes_genes, :] = 0

        if self.varm["refitted"].sum() == 0:  # if no gene can be refitted, we can skip
            return

        self.counts_to_refit = self.counts_to_refit[:, ~new_all_zeroes].copy()
        if isinstance(self.new_all_zeroes_genes, pd.MultiIndex):
            raise ValueError

        sub_dds = DeseqDataSet(
            counts=pd.DataFrame(
                self.counts_to_refit.X,
                index=self.counts_to_refit.obs_names,
                columns=self.counts_to_refit.var_names,
            ),
            metadata=self.obs,
            design_factors=self.design_formula,
            continuous_factors=self.continuous_factors,
            ref_level=self.ref_level,
            min_mu=self.min_mu,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            refit_cooks=self.refit_cooks,
            min_replicates=self.min_replicates,
            beta_tol=self.beta_tol,
            inference=self.inference,
        )

        # Use the same size factors
        sub_dds.obsm["size_factors"] = self.counts_to_refit.obsm["size_factors"]
        sub_dds.layers["normed_counts"] = (
            sub_dds.X / sub_dds.obsm["size_factors"][:, None]
        )

        # Estimate gene-wise dispersions.
        sub_dds.fit_genewise_dispersions()

        # Compute trend dispersions.
        # Note: the trend curve is not refitted.
        sub_dds.uns["disp_function_type"] = self.uns["disp_function_type"]
        if sub_dds.uns["disp_function_type"] == "parametric":
            sub_dds.uns["trend_coeffs"] = self.uns["trend_coeffs"]
        elif sub_dds.uns["disp_function_type"] == "mean":
            sub_dds.uns["mean_disp"] = self.uns["mean_disp"]
        sub_dds.varm["_normed_means"] = sub_dds.layers["normed_counts"].mean(0)
        # Reshape in case there's a single gene to refit
        sub_dds.varm["fitted_dispersions"] = sub_dds.disp_function(
            sub_dds.varm["_normed_means"]
        )

        # Estimate MAP dispersions.
        # Note: the prior variance is not recomputed.
        sub_dds.uns["_squared_logres"] = self.uns["_squared_logres"]
        sub_dds.uns["prior_disp_var"] = self.uns["prior_disp_var"]

        sub_dds.fit_MAP_dispersions()

        # Estimate log-fold changes (in natural log scale)
        sub_dds.fit_LFC()

        # Replace values in main object
        self.varm["_normed_means"][self.varm["refitted"]] = sub_dds.varm["_normed_means"]
        self.varm["LFC"][self.varm["refitted"]] = sub_dds.varm["LFC"]
        self.varm["genewise_dispersions"][self.varm["refitted"]] = sub_dds.varm[
            "genewise_dispersions"
        ]
        self.varm["fitted_dispersions"][self.varm["refitted"]] = sub_dds.varm[
            "fitted_dispersions"
        ]
        self.varm["dispersions"][self.varm["refitted"]] = sub_dds.varm["dispersions"]

        self.layers["replace_cooks"] = self.layers["cooks"].copy()

        for col in np.where(self.varm["refitted"])[0]:
            self.layers["replace_cooks"][self.obsm["replaceable"], col] = 0.0

    def _fit_iterate_size_factors(self, niter: int = 10, quant: float = 0.95) -> None:
        """
        Fit size factors using the ``iterative`` method.

        Used when each gene has at least one zero.

        Parameters
        ----------
        niter : int
            Maximum number of iterations to perform (default: ``10``).

        quant : float
            Quantile value at which negative likelihood is cut in the optimization
            (default: ``0.95``).

        """
        # Initialize size factors and normed counts fields
        self.obsm["size_factors"] = np.ones(self.n_obs)
        self.layers["normed_counts"] = self.X

        # Reduce the design matrix to an intercept and reconstruct at the end
        self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
        self.obsm["design_matrix"] = pd.DataFrame(
            1, index=self.obs_names, columns=["intercept"]
        )

        # Fit size factors using MLE
        def objective(p):
            sf = np.exp(p - np.mean(p))
            nll = nb_nll(
                counts=self[:, self.non_zero_genes].X,
                mu=self[:, self.non_zero_genes].layers["_mu_hat"]
                / self.obsm["size_factors"][:, None]
                * sf[:, None],
                alpha=self[:, self.non_zero_genes].varm["dispersions"],
            )
            # Take out the lowest likelihoods (highest neg) from the sum
            return np.sum(nll[nll < np.quantile(nll, quant)])

        for i in range(niter):
            # Estimate dispersions based on current size factors
            self.fit_genewise_dispersions()

            # Use a mean trend curve
            use_for_mean_genes = self.var_names[
                (self.varm["genewise_dispersions"] > 10 * self.min_disp)
                & self.varm["non_zero"]
            ]

            if len(use_for_mean_genes) == 0:
                print(
                    "No genes have a dispersion above 10 * min_disp in "
                    "_fit_iterate_size_factors."
                )
                break

            mean_disp = trim_mean(
                self[:, use_for_mean_genes].varm["genewise_dispersions"],
                proportiontocut=0.001,
            )

            self.varm["fitted_dispersions"] = np.ones(self.n_vars) * mean_disp
            self.fit_dispersion_prior()
            self.fit_MAP_dispersions()
            old_sf = self.obsm["size_factors"].copy()

            # Fit size factors using MLE
            res = minimize(objective, np.log(old_sf), method="Powell")

            self.obsm["size_factors"] = np.exp(res.x - np.mean(res.x))

            if not res.success:
                print("A size factor fitting iteration failed.", file=sys.stderr)
                break

            if (i > 1) and np.sum(
                (np.log(old_sf) - np.log(self.obsm["size_factors"])) ** 2
            ) < 1e-4:
                break
            elif i == niter - 1:
                print("Iterative size factor fitting did not converge.", file=sys.stderr)

        # Restore the design matrix and free buffer
        self.obsm["design_matrix"] = self.obsm["design_matrix_buffer"].copy()
        del self.obsm["design_matrix_buffer"]

        # Store normalized counts
        self.layers["normed_counts"] = self.X / self.obsm["size_factors"][:, None]

    def _check_full_rank_design(self):
        """Check that the design matrix has full column rank."""
        rank = np.linalg.matrix_rank(self.obsm["design_matrix"])
        num_vars = self.obsm["design_matrix"].shape[1]

        if len(self.obsm["design_matrix"]) != self.n_obs:
            raise ValueError("Design matrix and metadata have different lengths")

        if rank < num_vars:
            warnings.warn(
                "The design matrix is not full rank, so the model cannot be "
                "fitted, but some operations like design-free VST remain possible. "
                "To perform differential expression analysis, please remove the design "
                "variables that are linear combinations of others.",
                UserWarning,
                stacklevel=2,
            )
