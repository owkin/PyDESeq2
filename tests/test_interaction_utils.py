import copy

import pandas as pd

from pydeseq2.interaction_utils import build_single_interaction_factor


def test_build_single_factor():
    original_df = pd.DataFrame.from_dict(
        {"a": [1, 1, 2, 3, 2], "b": [4, 5, 6, 7, 8], "c": ["7", "8", "9", "10", "10"]}
    )
    design_factors = []
    # 2 interaction terms between continuous factors
    df = copy.deepcopy(original_df)
    build_single_interaction_factor(
        df, "a:b", design_factors, continuous_factors=["a", "b"]
    )
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the column multiplying both a and b coeff by coeff
    result["a:b"] = [a * b for a, b in zip(original_df["a"], original_df["b"])]
    pd.testing.assert_frame_equal(df, result)

    # 2 interacting terms between categorical factors
    df = copy.deepcopy(original_df)
    build_single_interaction_factor(df, "a:b", design_factors, continuous_factors=None)
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the column concatenating both a and  as str
    result["a:b"] = [str(a) + str(b) for a, b in zip(original_df["a"], original_df["b"])]
    pd.testing.assert_frame_equal(df, result)

    # 2 interacting terms with one categorical and one continuous column
    df = copy.deepcopy(original_df)
    build_single_interaction_factor(df, "a:b", design_factors, continuous_factors=["b"])
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the column multiplexing b to the categories of a
    zeros = [0.0, 0.0, 0.0, 0.0, 0.0]
    result["a_1:b"] = [
        z if idx not in [0, 1] else original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result["a_2:b"] = [
        z if idx not in [2, 4] else original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result["a_3:b"] = [
        z if idx != 3 else original_df["b"].iloc[idx] for idx, z in enumerate(zeros)
    ]
    pd.testing.assert_frame_equal(df, result)

    # 3 interacting terms with the last column being categorical
    df = copy.deepcopy(original_df)
    build_single_interaction_factor(
        df, "a:b:c", design_factors, continuous_factors=["a", "b"]
    )
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the following column multiplexe against the possible
    # values of c
    zeros = [0.0, 0.0, 0.0, 0.0, 0.0]
    result["a:b:c_7"] = [
        z if idx != 0 else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result["a:b:c_8"] = [
        z if idx != 1 else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result["a:b:c_9"] = [
        z if idx != 2 else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result["a:b:c_10"] = [
        z
        if idx not in [3, 4]
        else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    pd.testing.assert_frame_equal(df, result)

    # 3 interacting terms all categorical
    df = copy.deepcopy(original_df)
    build_single_interaction_factor(df, "a:b:c", design_factors, continuous_factors=None)
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the following column formed by concatenating
    # a, b, c as strings
    result["a:b:c"] = [
        str(a) + str(b) + str(c)
        for a, b, c in zip(original_df["a"], original_df["b"], original_df["c"])
    ]
    pd.testing.assert_frame_equal(df, result)

    # 3 interacting terms all continuous
    original_df["c"] = [float(el) for el in original_df["c"].tolist()]
    df = copy.deepcopy(original_df)
    original_continuous_factors = ["a", "b", "c"]
    continuous_factors = copy.deepcopy(original_continuous_factors)
    build_single_interaction_factor(
        df, "a:b:c", design_factors, continuous_factors=continuous_factors
    )

    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the following column formed by multiplying
    # a, b, c coeff by coeff
    result["a:b:c"] = [
        a * b * c
        for a, b, c in zip(original_df["a"], original_df["b"], original_df["c"])
    ]
    pd.testing.assert_frame_equal(df, result)
    # Make sure that we have not kept intermediate result in continuous factors
    assert set(continuous_factors) == set(original_continuous_factors + ["a:b:c"])
