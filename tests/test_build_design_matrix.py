import copy

import pandas as pd
from pandas import Categorical
from pandas import to_numeric

from pydeseq2.utils import build_design_matrix


def test_build_single_factor():
    original_df = pd.DataFrame.from_dict(
        {"a": [1, 1, 2, 3, 2], "b": [4, 5, 6, 7, 8], "c": ["7", "8", "9", "10", "10"]}
    )

    def tag_continuous_factors(df, continuous_factors=None):
        if continuous_factors is None:
            continuous_factors = []
        for factor in df.columns:
            if factor in continuous_factors:
                df[factor] = to_numeric(df[factor])
            else:
                df[factor] = Categorical(df[factor].astype("str"))

    # 1 interaction terms between continuous factors + all columns + all continuous
    df = copy.deepcopy(original_df)
    continuous_factors = ["a", "b", "c"]
    tag_continuous_factors(df, continuous_factors)

    design = build_design_matrix(df, "~ a+b +c+a:b")
    design.drop(columns=["intercept"], inplace=True)

    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the column multiplying both a and b coeff by coeff
    result["a:b"] = [a * b for a, b in zip(original_df["a"], original_df["b"])]
    tag_continuous_factors(result, continuous_factors)
    result["a:b"] = to_numeric(result["a:b"])

    pd.testing.assert_frame_equal(design, result)

    # 2 interacting terms between categorical factors wo other columuns
    df = copy.deepcopy(original_df)
    continuous_factors = None
    tag_continuous_factors(df, continuous_factors)
    design = build_design_matrix(df, "~ a:b")
    design.drop(columns=["intercept"], inplace=True)

    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the column concatenating both a and  as str
    result["a:b"] = [str(a) + str(b) for a, b in zip(original_df["a"], original_df["b"])]
    ref_levels = {
        "a": str(sorted(result["a"].unique())[0]),
        "b": str(sorted(result["b"].unique())[0]),
    }
    dummified_b = pd.get_dummies(result["b"], prefix="b", drop_first=True).astype("int")
    dummified_b.rename(
        columns={col: col + "_vs_" + ref_levels["b"] for col in dummified_b.columns},
        inplace=True,
    )
    dummified_ab = pd.get_dummies(result["a:b"], drop_first=True).astype("int")
    dummified_ab.rename(
        columns={
            k: "a:b_" + k[0] + k[1] + "_vs_" + ref_levels["a"] + ref_levels["b"]
            for k in dummified_ab.columns
        },
        inplace=True,
    )
    # formulaic also encodes all not found combinations of a and b as long as
    # it's full-rank
    import itertools

    ab_list = itertools.product(result["a"].unique(), result["b"].unique())
    for ab in ab_list:
        ab_str = (str(ab[0]), str(ab[1]))
        if "".join(ab_str) not in result["a:b"].unique():
            dummified_ab[
                "a:b_"
                + ab_str[0]
                + ab_str[1]
                + "_vs_"
                + ref_levels["a"]
                + ref_levels["b"]
            ] = 0
    # Now we can drop some columns because of formulaic's complicated full-rankness
    # matric construction
    result = pd.concat([dummified_b, dummified_ab], axis=1)[design.columns]

    pd.testing.assert_frame_equal(design, result)

    # 3 interacting terms with one categorical and one continuous column and no
    # other column
    df = copy.deepcopy(original_df)
    continuous_factors = ["b"]
    tag_continuous_factors(df, continuous_factors)
    design = build_design_matrix(
        df,
        "~a:b",
    )
    design.drop(columns=["intercept"], inplace=True)
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the column multiplexing b to the categories of a
    zeros = [0.0, 0.0, 0.0, 0.0, 0.0]
    ref_levels = {"a": str(sorted(result["a"].unique())[0])}
    result[f"a:b_1_vs_{ref_levels['a']}"] = [
        int(z) if idx not in [0, 1] else int(original_df["b"].iloc[idx])
        for idx, z in enumerate(zeros)
    ]
    result[f"a:b_2_vs_{ref_levels['a']}"] = [
        int(z) if idx not in [2, 4] else int(original_df["b"].iloc[idx])
        for idx, z in enumerate(zeros)
    ]
    result[f"a:b_3_vs_{ref_levels['a']}"] = [
        int(z) if idx != 3 else int(original_df["b"].iloc[idx])
        for idx, z in enumerate(zeros)
    ]
    # full-rankedness

    result = result[design.columns]
    pd.testing.assert_frame_equal(design, result)

    # 4, 3(!) interacting terms with the last one being categorical
    df = copy.deepcopy(original_df)
    continuous_factors = ["a", "b"]
    tag_continuous_factors(df, continuous_factors)
    design = build_design_matrix(
        df,
        "~a:b:c",
    )
    design.drop(columns=["intercept"], inplace=True)

    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the following column multiplexed against the possible
    # values of c
    zeros = [0.0, 0.0, 0.0, 0.0, 0.0]
    ref_levels = {"c": str(sorted(result["c"].unique())[0])}
    result[f"a:b:c_7_vs_{ref_levels['c']}"] = [
        z if idx != 0 else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result[f"a:b:c_8_vs_{ref_levels['c']}"] = [
        z if idx != 1 else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result[f"a:b:c_9_vs_{ref_levels['c']}"] = [
        z if idx != 2 else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        for idx, z in enumerate(zeros)
    ]
    result[f"a:b:c_10_vs_{ref_levels['c']}"] = [
        (
            z
            if idx not in [3, 4]
            else original_df["a"].iloc[idx] * original_df["b"].iloc[idx]
        )
        for idx, z in enumerate(zeros)
    ]
    triple_interaction_columns = [col for col in result.columns if "a:b:c" in col]
    # This is done in formulaic when hybrid interactions are present
    for col in triple_interaction_columns:
        result[col] = result[col].astype("int64")

    pd.testing.assert_frame_equal(design, result[design.columns])

    # 5, 3(!) interacting terms all categorical
    df = copy.deepcopy(original_df)
    continuous_factors = None
    tag_continuous_factors(df, continuous_factors)
    design = build_design_matrix(
        df,
        "~a:b:c",
    )
    design.drop(columns=["intercept"], inplace=True)
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the following column formed by concatenating
    # a, b, c as strings
    result["a:b:c"] = [
        str(a) + str(b) + str(c)
        for a, b, c in zip(original_df["a"], original_df["b"], original_df["c"])
    ]
    ref_levels = {
        "a": str(sorted(result["a"].unique())[0]),
        "b": str(sorted(result["b"].unique())[0]),
        "c": str(sorted(result["c"].unique())[0]),
    }
    dummified_abc = pd.get_dummies(
        result["a:b:c"], prefix="a:b:c", drop_first=True
    ).astype("int")
    dummified_abc.rename(
        columns={
            col: col + "_vs_" + ref_levels["a"] + ref_levels["b"] + ref_levels["c"]
            for col in dummified_abc.columns
        },
        inplace=True,
    )
    import itertools

    abc_list = itertools.product(
        result["a"].unique(), result["b"].unique(), result["c"].unique()
    )
    for abc in abc_list:
        abc_str = (str(abc[0]), str(abc[1]), str(abc[2]))
        if "".join(abc_str) not in result["a:b:c"].unique():
            dummified_abc[
                "a:b:c_"
                + abc_str[0]
                + abc_str[1]
                + abc_str[2]
                + "_vs_"
                + ref_levels["a"]
                + ref_levels["b"]
                + ref_levels["c"]
            ] = 0
    # For simplicity we only check triple interacting terms but giving a:b:c to
    # formulaic also forms all double products a:b, a:c, b:c modulo rank
    # TODO add tests for doubly interacting terms
    triple_interaction_columns = [col for col in design.columns if "a:b:c" in col]

    pd.testing.assert_frame_equal(
        design[triple_interaction_columns], dummified_abc[triple_interaction_columns]
    )

    # 6, 3(!) interacting terms all continuous
    original_df["c"] = [float(el) for el in original_df["c"].tolist()]
    df = copy.deepcopy(original_df)
    original_continuous_factors = ["a", "b", "c"]
    continuous_factors = copy.deepcopy(original_continuous_factors)

    tag_continuous_factors(df, continuous_factors)
    design = build_design_matrix(df, "~a:b:c")
    design.drop(columns=["intercept"], inplace=True)
    # The result should be the original df
    result = copy.deepcopy(original_df)
    # to which we have added the following column formed by multiplying
    # a, b, c coeff by coeff
    result["a:b:c"] = [
        a * b * c
        for a, b, c in zip(original_df["a"], original_df["b"], original_df["c"])
    ]
    # For simplicity we only check triple interacting terms but giving a:b:c to
    # formulaic also forms all double products a:b, a:c, b:c modulo rank
    # TODO add tests for doubly interacting terms
    triple_interaction_columns = [col for col in design.columns if "a:b:c" in col]
    pd.testing.assert_frame_equal(
        design[triple_interaction_columns], result[triple_interaction_columns]
    )
