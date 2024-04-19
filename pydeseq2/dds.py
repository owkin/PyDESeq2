from __future__ import annotations

import copy
import itertools
import re
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

    ref_level : list or None
        An optional list of tuples each with two strings of the form
        ``("factor", "test_level")`` specifying the factor of interest and the
        reference (control) level against which we're testing, e.g.
        ``[("condition", "A")]``. (default: ``None``).

    trend_fit_type : str
        Either "parametric" or "mean" for the type of fitting of the dispersions
        trend curve. (default: ``"parametric"``).
    design_matrix: pd.DataFrame or None
        If given will take precedence over design_factors and ref_level.
        (default: ``None``).

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
        available cpus will be used by the ``DefaultInference``. If both are specified,
        it will try to override the ``n_cpus`` attribute of the ``inference`` object.
        (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).

    quiet : bool
        Suppress deseq2 status updates during fit.

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

    fit_type: str
        Either "parametric" or "mean" for the type of fitting of dispersions to the
        mean intensity. "parametric": fit a dispersion-mean relation via a robust
        gamma-family GLM. "mean": use the mean of gene-wise dispersion estimates.
        (default: ``"parametric"``).

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
        trend_fit_type: Literal["parametric", "mean"] = "parametric",
        ref_level: list[tuple[str, str]] | None = None,
        design_matrix: pd.DataFrame | None = None,
        min_mu: float = 0.5,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        refit_cooks: bool = True,
        min_replicates: int = 7,
        beta_tol: float = 1e-8,
        n_cpus: int | None = None,
        inference: Inference | None = None,
        quiet: bool = False,
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

        self.design_factors = design_factors
        self.continuous_factors = (
            continuous_factors if continuous_factors is not None else []
        )
        # If it's a list we convert it right back to a formulaic string
        if isinstance(self.design_factors, list):
            # Little bit of careful formating as below to remove space around : symbols
            self.design_factors = [
                re.sub(r"(?:(?<=\:) +| +(?=\:))", "", factor)
                for factor in self.design_factors
            ]
            # We extract all single factors
            extracted_single_factors = list(
                itertools.chain(*[factor.split(":") for factor in self.design_factors])
            )

            self.single_design_factors = extracted_single_factors
            # We store design factors as a list for later reuse in contrast
            self.design_factors_list = copy.deepcopy(self.design_factors)
            self.design_factors = "~" + "+".join(self.design_factors)

        # Following lines handle the case where the user provides a formula
        # of the type formulaic can recognize or a column everything will be
        # converted to formulaic formulas as in the first case
        elif isinstance(self.design_factors, str):
            # TODO uncomment when we support all of formulaic grammar
            # try:
            #     Formula(self.design_factors)
            # except FormulaInvalidError as err:
            #     raise (
            #         f"Design factor {self.design_factors} is not a valid formula"
            #     ) from err
            # Remove potential spaces around +,: or ~ signs for easier parsing
            # of individual factor
            if not self.design_factors.startswith("~"):
                self.design_factors = "~" + self.design_factors

            self.design_factors = re.sub(
                r"(?:(?<=\+) +| +(?=\+))", "", self.design_factors
            )
            self.design_factors = re.sub(
                r"(?:(?<=\:) +| +(?=\:))", "", self.design_factors
            )
            self.design_factors = re.sub(
                r"(?:(?<=\~) +| +(?=\~))", "", self.design_factors
            )
            # Now we should have something of the form '~a+b+a:b:c:d'
            # We also support the fact of passing only one factor such as 'a' by
            # converting it to the corresponding formula '~a'
            # Thanks to the previous reformatting we can do easily single-factor
            # extraction
            extracted_single_factors = list(
                set(
                    itertools.chain(
                        *[term.split(":") for term in self.design_factors[1:].split("+")]
                    )
                )
            )
            # Remove non unique characters while keeping the apparition order
            factors_dict: dict[str, None] = {}
            for factor in extracted_single_factors:
                factors_dict[factor] = None
            extracted_single_factors = list(factors_dict.keys())
            self.single_design_factors = extracted_single_factors

        else:
            raise ValueError(
                "Design factors should be a string or a list of strings got"
                f" {type(self.design_factors)}"
            )

        # We check all single factors extracted were indeed in the columns
        for factor in self.single_design_factors:
            if factor not in self.obs.columns:
                raise ValueError(f"Formula contain unknown extracted factor {factor}")

        # Check that design factors don't contain underscores. If so, convert them to
        # hyphens.
        # We modify only obs object
        if np.any(["_" in factor for factor in self.single_design_factors]):
            warnings.warn(
                """Some factor names in the design contain underscores ('_').
                They will be converted to hyphens ('-').""",
                UserWarning,
                stacklevel=2,
            )
            new_factors = replace_underscores(self.single_design_factors)
            self.obs.rename(
                columns=dict(zip(self.single_design_factors, new_factors)),
                inplace=True,
            )
            self.single_design_factors = new_factors
            # Also check continuous factors
            if continuous_factors is not None:
                self.continuous_factors = replace_underscores(continuous_factors)

        # If ref_level has underscores, convert them to hyphens
        # Don't raise a warning: it will be raised by build_design_matrix()
        if ref_level is not None:
            ref_level = replace_underscores(ref_level)

        self.design_factors = replace_underscores(self.design_factors)
        # We need it for the contrast
        assert isinstance(self.design_factors, str)
        self.design_factors_list = self.design_factors[1:].split("+")

        if self.obs[self.single_design_factors].isna().any().any():
            raise ValueError("NaNs are not allowed in the design factors.")

        # We convert every factor in the design_factors to string type
        assert isinstance(
            self.single_design_factors, list
        ), "Could not extract any single factor"
        for factor in self.single_design_factors:
            if factor not in self.continuous_factors:
                self.obs[factor] = Categorical(self.obs[factor].astype(str))
                levels = self.obs[factor].unique()
                if "_" in "".join(levels):
                    warnings.warn(
                        f"""Some category in the design of factor {factor} contain
                        underscores ('_'). They will be converted to hyphens ('-').""",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.obs[factor] = self.obs[factor].cat.rename_categories(
                        replace_underscores(levels.to_list())
                    )

            else:
                self.obs[factor] = to_numeric(self.obs[factor])

        # Build the design matrix or use given one
        # Stored in the obsm attribute of the dataset
        if design_matrix is None:
            assert isinstance(
                self.design_factors, str
            ), "By now design_factors should be a string"
            self.obsm["design_matrix"] = build_design_matrix(
                metadata=self.obs,
                design_factors=self.design_factors,
                ref_level=ref_level,
            )
            # Now tthat formulaic preprocessed the formula we need it as a list
            self.design_factors = self.design_factors_list
        else:
            warnings.warn(
                """Design matrix was given; ignoring design_factors, continuous factors
                and ref_level""",
                UserWarning,
                stacklevel=2,
            )
            if len(design_matrix.index) != len(self.obs.index):
                raise ValueError("Design matrix and metadata have different lengths")
            if "intercept" not in design_matrix.columns:
                warnings.warn(
                    "Careful provided design matrix doesn't have intercept",
                    UserWarning,
                    stacklevel=2,
                )

            if not design_matrix["intercept"].equals(
                pd.Series(1.0, index=design_matrix.index)
            ):
                warnings.warn(
                    "'intercept' column is not all ones.", UserWarning, stacklevel=2
                )

            def col_name_parser(colname, split_interactions=True):
                # There can be either 0 or one vs
                colname = colname.split("_vs_")
                assert len(colname) in {1, 2}, "There should be 0 or 1 _vs_ in the name"
                if len(colname) == 1:
                    return [colname[0]]
                else:
                    # Left part before the underscore is an interaction term
                    name_with_interaction = colname[0].split("_")[0]
                    if not split_interactions:
                        return name_with_interaction
                    else:
                        return name_with_interaction.split(":")

            extracted_single_factors = list(
                itertools.chain(
                    *[
                        col_name_parser(col)
                        for col in design_matrix
                        if col != "intercept"
                    ]
                )
            )
            # Remove non unique characters while keeping the apparition orde
            factors_dict = {}
            for factor in extracted_single_factors:
                factors_dict[factor] = None
            extracted_single_factors = list(factors_dict.keys())

            # We extract all individual factors from the design matrix following
            # pydeseq2 naming conventions containing potentially _vs_
            for individual_factor in extracted_single_factors:
                if individual_factor not in self.obs.columns:
                    raise ValueError(
                        "Design matrix contains unknown factor, be careful if"
                        " there were underscores in the column names they were"
                        " converted automatically to underscores"
                        " in the obs matrix."
                        f"Could not match extracted factor: {individual_factor}"
                        f" with the list of factors found in the observation:"
                        f" {self.obs.columns}, make sure the name of the columns"
                        " of your design matrix respects pydeseq2 conventions:"
                        "colname = 'factor1:...:factorN_val1...val3_vs_ref1...ref2ref3'"
                    )

            # With some luck here design_matrix is valid all the time
            self.obsm["design_matrix"] = design_matrix

            self.single_design_factors = extracted_single_factors

            # We store design factors as a list for later reuse in contrast
            self.design_factors = [
                col_name_parser(col, split_interactions=False)
                for col in design_matrix
                if col != "intercept"
            ]

        if len(self.single_design_factors) == 1:
            if self.obs[self.single_design_factors[0]].nunique() == 1:
                raise ValueError("Only one unique value and one design factor.")

        # Check that the design matrix has full rank
        self._check_full_rank_design()

        self.trend_fit_type = trend_fit_type
        self.min_mu = min_mu
        self.min_disp = min_disp
        self.max_disp = np.maximum(max_disp, self.n_obs)
        self.refit_cooks = refit_cooks
        self.ref_level = ref_level
        self.min_replicates = min_replicates
        self.beta_tol = beta_tol
        self.quiet = quiet
        self.logmeans = None
        self.filtered_genes = None

        if inference:
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

    def vst(
        self,
        use_design: bool = False,
        fit_type: Literal["parametric", "mean"] = "parametric",
    ) -> None:
        """Fit a variance stabilizing transformation, and apply it to normalized counts.

        Results are stored in ``dds.layers["vst_counts"]``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used. (default: ``False``).
        fit_type: str
            Either "parametric" or "mean" for the type of fitting of dispersions to the
            mean intensity. "parametric": fit a dispersion-mean relation via a robust
            gamma-family GLM. "mean": use the mean of gene-wise dispersion estimates.
            (default: ``"parametric"``).
        """
        self.vst_fit(use_design=use_design, fit_type=fit_type)
        self.layers["vst_counts"] = self.vst_transform()

    def vst_fit(
        self,
        use_design: bool = False,
        fit_type: Literal["parametric", "mean"] = "parametric",
    ) -> None:
        """Fit a variance stabilizing transformation.

        Results are stored in ``dds.layers["vst_counts"]``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used. (default: ``False``).
        fit_type : str
            Either "parametric" or "mean" for the type of fitting of dispersions to the
            mean intensity. parametric - fit a dispersion-mean relation via a robust
            gamma-family GLM. mean - use the mean of gene-wise dispersion estimates.
            (default: ``"parametric"``).
        """
        self.fit_type = fit_type  # to re-use inside vst_transform

        # Start by fitting median-of-ratio size factors if not already present,
        # or if they were computed iteratively
        if "size_factors" not in self.obsm or self.logmeans is None:
            self.fit_size_factors()  # by default, fit_type != "iterative"

        if use_design:
            # Check that the dispersion trend curve was fitted. If not, fit it.
            # This will call previous functions in a cascade.
            if "disp_function" not in self.uns:
                self.fit_dispersion_trend()
        else:
            # Reduce the design matrix to an intercept and reconstruct at the end
            self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
            self.obsm["design_matrix"] = pd.DataFrame(
                1, index=self.obs_names, columns=[["intercept"]]
            )
            # Fit the trend curve with an intercept design
            self.fit_genewise_dispersions()
            if self.fit_type == "parametric":
                self.fit_dispersion_trend()

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

        if self.fit_type == "parametric":
            a0, a1 = self.uns["trend_coeffs"]
            return np.log2(
                (
                    1
                    + a1
                    + 2 * a0 * normed_counts
                    + 2 * np.sqrt(a0 * normed_counts * (1 + a1 + a0 * normed_counts))
                )
                / (4 * a0)
            )
        elif self.fit_type == "mean":
            gene_dispersions = self.varm["genewise_dispersions"]
            use_for_mean = gene_dispersions > 10 * self.min_disp
            mean_disp = trim_mean(gene_dispersions[use_for_mean], proportiontocut=0.001)
            return (
                2 * np.arcsinh(np.sqrt(mean_disp * normed_counts))
                - np.log(mean_disp)
                - np.log(4)
            ) / np.log(2)
        else:
            raise NotImplementedError(
                f"Found fit_type '{self.fit_type}'. Expected 'parametric' or 'mean'."
            )

    def deseq2(self) -> None:
        """Perform dispersion and log fold-change (LFC) estimation.

        Wrapper for the first part of the PyDESeq2 pipeline.
        """
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

    def fit_size_factors(
        self, fit_type: Literal["ratio", "iterative"] = "ratio"
    ) -> None:
        """Fit sample-wise deseq2 normalization (size) factors.

        Uses the median-of-ratios method: see :func:`pydeseq2.preprocessing.deseq2_norm`,
        unless each gene has at least one sample with zero read counts, in which case it
        switches to the ``iterative`` method.

        Parameters
        ----------
        fit_type : str
            The normalization method to use (default: ``"ratio"``).
        """
        if not self.quiet:
            print("Fitting size factors...", file=sys.stderr)
        start = time.time()
        if fit_type == "iterative":
            self._fit_iterate_size_factors()
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
            (
                self.layers["normed_counts"],
                self.obsm["size_factors"],
            ) = deseq2_norm_transform(self.X, self.logmeans, self.filtered_genes)
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

    def fit_genewise_dispersions(self) -> None:
        """Fit gene-wise dispersion estimates.

        Fits a negative binomial per gene, independently.
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

        self.layers["_mu_hat"] = np.full((self.n_obs, self.n_vars), np.NaN)
        self.layers["_mu_hat"][:, self.varm["non_zero"]] = mu_hat_

        if not self.quiet:
            print("Fitting dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=self.X[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers["_mu_hat"][:, self.non_zero_idx],
            alpha_hat=self.varm["_MoM_dispersions"][self.non_zero_idx],
            min_disp=self.min_disp,
            max_disp=self.max_disp,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.varm["genewise_dispersions"] = np.full(self.n_vars, np.NaN)
        self.varm["genewise_dispersions"][self.varm["non_zero"]] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_genewise_converged"] = np.full(self.n_vars, np.NaN)
        self.varm["_genewise_converged"][self.varm["non_zero"]] = l_bfgs_b_converged_

    def fit_dispersion_trend(self) -> None:
        """Fit the dispersion trend curve.

        The type of the trend curve is determined by ``self.trend_fit_type``.
        """
        # Check that genewise dispersions are available. If not, compute them.
        if "genewise_dispersions" not in self.varm:
            self.fit_genewise_dispersions()

        if not self.quiet:
            print("Fitting dispersion trend curve...", file=sys.stderr)
        start = time.time()
        self.varm["_normed_means"] = self.layers["normed_counts"].mean(0)

        if self.trend_fit_type == "parametric":
            self._fit_parametric_dispersion_trend()
        elif self.trend_fit_type == "mean":
            self._fit_mean_dispersion_trend()
        else:
            raise NotImplementedError(
                f"Expected 'parametric' or 'meand' trend curve fit "
                f"types, received {self.trend_fit_type}"
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

        self.varm["MAP_dispersions"] = np.full(self.n_vars, np.NaN)
        self.varm["MAP_dispersions"][self.varm["non_zero"]] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_MAP_converged"] = np.full(self.n_vars, np.NaN)
        self.varm["_MAP_converged"][self.varm["non_zero"]] = l_bfgs_b_converged_

        # Filter outlier genes for which we won't apply shrinkage
        self.varm["dispersions"] = self.varm["MAP_dispersions"].copy()
        self.varm["_outlier_genes"] = np.log(self.varm["genewise_dispersions"]) > np.log(
            self.varm["fitted_dispersions"]
        ) + 2 * np.sqrt(self.uns["_squared_logres"])
        self.varm["dispersions"][self.varm["_outlier_genes"]] = self.varm[
            "genewise_dispersions"
        ][self.varm["_outlier_genes"]]

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
            np.NaN,
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

        self.layers["_mu_LFC"] = np.full((self.n_obs, self.n_vars), np.NaN)
        self.layers["_mu_LFC"][:, self.varm["non_zero"]] = mu_

        self.layers["_hat_diagonals"] = np.full((self.n_obs, self.n_vars), np.NaN)
        self.layers["_hat_diagonals"][:, self.varm["non_zero"]] = hat_diagonals_

        self.varm["_LFC_converged"] = np.full(self.n_vars, np.NaN)
        self.varm["_LFC_converged"][self.varm["non_zero"]] = converged_

    def calculate_cooks(self) -> None:
        """Compute Cook's distance for outlier detection.

        Measures the contribution of a single entry to the output of LFC estimation.
        """
        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()

        num_vars = self.obsm["design_matrix"].shape[-1]

        # Keep only non-zero genes
        nonzero_data = self[:, self.non_zero_genes]
        normed_counts = pd.DataFrame(
            nonzero_data.X / self.obsm["size_factors"][:, None],
            index=self.obs_names,
            columns=self.non_zero_genes,
        )

        dispersions = robust_method_of_moments_disp(
            normed_counts, self.obsm["design_matrix"]
        )

        V = (
            nonzero_data.layers["_mu_LFC"]
            + dispersions.values[None, :] * nonzero_data.layers["_mu_LFC"] ** 2
        )
        squared_pearson_res = (nonzero_data.X - nonzero_data.layers["_mu_LFC"]) ** 2 / V
        diag_mul = (
            nonzero_data.layers["_hat_diagonals"]
            / (1 - nonzero_data.layers["_hat_diagonals"]) ** 2
        )

        self.layers["cooks"] = np.full((self.n_obs, self.n_vars), np.NaN)
        self.layers["cooks"][:, self.varm["non_zero"]] = (
            squared_pearson_res / num_vars * diag_mul
        )

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

        self.varm["_MoM_dispersions"] = np.full(self.n_vars, np.NaN)
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

    def _fit_parametric_dispersion_trend(self):
        r"""Fit the dispersion curve according to a parametric model.

        :math:`f(\mu) = \alpha_1/\mu + a_0`.
        """
        # Exclude all-zero counts
        targets = pd.Series(
            self[:, self.non_zero_genes].varm["genewise_dispersions"].copy(),
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

                self._fit_mean_dispersion_trend()
                return

            # Filter out genes that are too far away from the curve before refitting
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
        self.uns["trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])

        self.varm["fitted_dispersions"] = np.full(self.n_vars, np.NaN)
        self.uns["disp_function_type"] = "parametric"
        self.varm["fitted_dispersions"][self.varm["non_zero"]] = self.disp_function(
            self.varm["_normed_means"][self.varm["non_zero"]]
        )

    def _fit_mean_dispersion_trend(self):
        """Use the mean of dispersions as trend curve."""
        self.uns["mean_disp"] = trim_mean(
            self.varm["genewise_dispersions"][
                self.varm["genewise_dispersions"] > 10 * self.min_disp
            ],
            proportiontocut=0.001,
        )

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
            self.varm["replaced"] = pd.Series(False, index=self.var_names)
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
            design_factors=self.design_factors,
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
        self.varm["dispersions"][self.varm["refitted"]] = sub_dds.varm["dispersions"]

        replace_cooks = pd.DataFrame(self.layers["cooks"].copy())
        replace_cooks.loc[self.obsm["replaceable"], self.varm["refitted"]] = 0.0

        self.layers["replace_cooks"] = replace_cooks

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
            1, index=self.obs_names, columns=[["intercept"]]
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

        if rank < num_vars:
            warnings.warn(
                "The design matrix is not full rank, so the model cannot be "
                "fitted, but some operations like design-free VST remain possible. "
                "To perform differential expression analysis, please remove the design "
                "variables that are linear combinations of others.",
                UserWarning,
                stacklevel=2,
            )
