import numpy as np
import pandas as pd
import pytest

from pydeseq2.dds import DeseqDataSet
from pydeseq2.utils import build_design_matrix
from pydeseq2.utils import load_example_data


@pytest.mark.parametrize(
    "design_factors",
    [
        ["condition"],
        ["condition", "group"],
        ["condition:group"],
        ["condition", "group", "condition:group"],
        "~condition:group",
        "~condition+condition:group",
    ],
)
def test_dds_with_factors(design_factors):
    """Test that inputting factor works"""

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )
    # Create dataset
    DeseqDataSet(counts=counts_df, metadata=metadata, design_factors=design_factors)


@pytest.mark.parametrize(
    "design_matrix",
    [
        pd.DataFrame(),
        pd.DataFrame.from_dict(
            {
                "intercept": (2 * np.ones(100)).tolist(),
                "condition": np.ones(50).tolist() + np.zeros(50).tolist(),
            }
        ),
        pd.DataFrame.from_dict(
            {
                "intercept": (2 * np.ones(100)).tolist(),
                "condition1": np.ones(50).tolist() + np.zeros(50).tolist(),
            }
        ),
        "regular",
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dds_with_design_matrix(design_matrix):
    """Test that inputting factor works"""

    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    metadata = load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )
    if isinstance(design_matrix, str):
        if design_matrix == "regular":
            design_matrix = build_design_matrix(
                metadata=metadata,
                design_factors="~condition+condition:group",
                intercept=True,
            )

        # Create dataset
        DeseqDataSet(counts=counts_df, metadata=metadata, design_matrix=design_matrix)
    # make sure it raises an error
    else:
        with pytest.raises(ValueError):
            DeseqDataSet(
                counts=counts_df, metadata=metadata, design_matrix=design_matrix
            )
