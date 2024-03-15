import pytest

from pydeseq2.dds import DeseqDataSet
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
