"""
Copied from pertpy
https://github.com/scverse/pertpy/tests/tools/_differential_gene_expression/
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from formulaic.parser.types import Factor

from pydeseq2._formulaic import AmbiguousAttributeError
from pydeseq2._formulaic import FactorMetadata
from pydeseq2._formulaic import get_factor_storage_and_materializer
from pydeseq2._formulaic import resolve_ambiguous
from pydeseq2.utils import load_example_data


@pytest.fixture
def test_counts():
    return load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )


@pytest.fixture
def test_metadata():
    return load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )


@pytest.fixture
def test_adata(test_counts, test_metadata):
    return ad.AnnData(X=test_counts, obs=test_metadata)


@pytest.fixture(params=[np.array, sp.csr_matrix, sp.csc_matrix])
def test_adata_minimal(request):
    matrix_format = request.param
    n_obs = 80
    n_donors = n_obs // 4
    rng = np.random.default_rng(9)  # make tests deterministic
    obs = pd.DataFrame(
        {
            "condition": ["A", "B"] * (n_obs // 2),
            "donor": sum(([f"D{i}"] * n_donors for i in range(n_obs // n_donors)), []),
            "other": (["X"] * (n_obs // 4)) + (["Y"] * ((3 * n_obs) // 4)),
            "pairing": sum(([str(i), str(i)] for i in range(n_obs // 2)), []),
            "continuous": [rng.uniform(0, 1) * 4000 for _ in range(n_obs)],
        },
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    group1 = rng.negative_binomial(20, 0.1, n_obs // 2)  # large mean
    group2 = rng.negative_binomial(5, 0.5, n_obs // 2)  # small mean

    condition_data = np.empty((n_obs,), dtype=group1.dtype)
    condition_data[0::2] = group1
    condition_data[1::2] = group2

    donor_data = np.empty((n_obs,), dtype=group1.dtype)
    donor_data[0:n_donors] = group2[:n_donors]
    donor_data[n_donors : (2 * n_donors)] = group1[n_donors:]

    donor_data[(2 * n_donors) : (3 * n_donors)] = group2[:n_donors]
    donor_data[(3 * n_donors) :] = group1[n_donors:]

    X = matrix_format(np.vstack([condition_data, donor_data]).T)

    return ad.AnnData(X=X, obs=obs, var=var)


# Ignore anndata ImplicitModificationWarning
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "formula,reorder_categorical,expected_factor_metadata",
    [
        [
            "~ donor",
            None,
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"}},
        ],
        [
            "~ donor",
            {"donor": ["D2", "D1", "D0", "D3"]},
            {"donor": {"reduced_rank": True, "custom_encoder": False, "base": "D2"}},
        ],
        [
            "~ C(donor)",
            None,
            {"C(donor)": {"reduced_rank": True, "custom_encoder": True, "base": "D0"}},
        ],
        [
            "~ C(donor, contr.treatment(base='D2'))",
            None,
            {
                "C(donor, contr.treatment(base='D2'))": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "D2",
                }
            },
        ],
        [
            "~ C(donor, contr.sum)",
            None,
            {
                "C(donor, contr.sum)": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "D3",
                }
            },
        ],
        [
            "~ C(donor, contr.sum)",
            {"donor": ["D1", "D0", "D3", "D2"]},
            {
                "C(donor, contr.sum)": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "D2",
                }
            },
        ],
        [
            "~ condition",
            None,
            {"condition": {"reduced_rank": True, "custom_encoder": False, "base": "A"}},
        ],
        [
            "~ C(condition)",
            None,
            {
                "C(condition)": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "A",
                }
            },
        ],
        [
            "~ C(condition, contr.treatment(base='B'))",
            None,
            {
                "C(condition, contr.treatment(base='B'))": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "B",
                }
            },
        ],
        [
            "~ C(condition, contr.sum)",
            None,
            {
                "C(condition, contr.sum)": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "B",
                }
            },
        ],
        [
            "~ 0 + condition",
            None,
            {
                "condition": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                }
            },
        ],
        [
            "~ condition + donor",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                },
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
            },
        ],
        [
            "~ 0 + condition + donor",
            None,
            {
                "condition": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                },
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
            },
        ],
        [
            "~ condition * donor",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                },
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
            },
        ],
        [
            "~ condition * C(donor, contr.treatment(base='D2'))",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                },
                "C(donor, contr.treatment(base='D2'))": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "D2",
                },
            },
        ],
        [
            "~ condition + C(condition) + C(condition, contr.treatment(base='B'))",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                },
                "C(condition)": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "A",
                },
                "C(condition, contr.treatment(base='B'))": {
                    "reduced_rank": True,
                    "custom_encoder": True,
                    "base": "B",
                },
            },
        ],
        [
            "~ condition + continuous + np.log(continuous)",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                    "kind": Factor.Kind.CATEGORICAL,
                },
                "continuous": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                    "kind": Factor.Kind.NUMERICAL,
                },
                "np.log(continuous)": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                    "kind": Factor.Kind.NUMERICAL,
                },
            },
        ],
        [
            "~ condition * donor + continuous",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                },
                "donor": {"reduced_rank": True, "custom_encoder": False, "base": "D0"},
                "continuous": {
                    "reduced_rank": False,
                    "custom_encoder": False,
                    "base": None,
                    "kind": Factor.Kind.NUMERICAL,
                },
            },
        ],
        [
            "~ condition:donor",
            None,
            {
                "condition": {
                    "reduced_rank": True,
                    "custom_encoder": False,
                    "base": "A",
                },
                "donor": {
                    "custom_encoder": False,
                    "drop_field": "D0",
                },  # `reduced_rank` and `base` will be ambigous here because Formulaic
                # generates both version of the factor internally
            },
        ],
    ],
)
def test_custom_materializer(
    test_adata_minimal, formula, reorder_categorical, expected_factor_metadata
):
    """Test that the custom materializer correctly stores the baseline category.

    Parameters
    ----------
    test_adata_minimal
        adata fixture
    formula
        Formula to test
    reorder_categorical
        Create a pandas categorical for a given column with a certain order of categories
    expected_factor_metadata
        dict with expected values for each factor
    """
    if reorder_categorical is not None:
        for col, order in reorder_categorical.items():
            test_adata_minimal.obs[col] = pd.Categorical(
                test_adata_minimal.obs[col], categories=order
            )
    factor_storage, _, materializer = get_factor_storage_and_materializer()
    materializer(test_adata_minimal.obs, record_factor_metadata=True).get_model_matrix(
        formula
    )
    for factor, expected_metadata in expected_factor_metadata.items():
        actual_metadata = factor_storage[factor]
        for k in expected_metadata:
            assert resolve_ambiguous(actual_metadata, k) == expected_metadata[k]


# Ignore anndata ImplicitModificationWarning
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_resolve_ambiguous():
    obj1 = FactorMetadata("F1", True, True, ["A", "B"], Factor.Kind.CATEGORICAL)
    obj2 = FactorMetadata("F2", True, False, ["A", "B"], Factor.Kind.CATEGORICAL)
    obj3 = FactorMetadata("F3", True, False, None, Factor.Kind.NUMERICAL)

    with pytest.raises(ValueError):
        resolve_ambiguous([], "foo")

    with pytest.raises(AttributeError):
        resolve_ambiguous([obj1, obj2], "doesntexist")

    with pytest.raises(AmbiguousAttributeError):
        assert resolve_ambiguous([obj1, obj2], "name")

    assert resolve_ambiguous([obj1, obj2, obj3], "reduced_rank") is True
    assert resolve_ambiguous([obj1, obj2], "categories") == ["A", "B"]

    with pytest.raises(AmbiguousAttributeError):
        assert resolve_ambiguous([obj1, obj2, obj3], "categories")

    with pytest.raises(AmbiguousAttributeError):
        assert resolve_ambiguous([obj1, obj3], "kind")
