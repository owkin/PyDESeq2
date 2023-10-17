import shutil
from pathlib import Path
from typing import Union
from urllib.request import urlopen

import pandas as pd

_URL = "http://genomedata.org/gen-viz-workshop/intro_to_deseq2/tutorial/E-GEOD-50760-raw-counts.tsv"  # noqa
_RAW_COUNTS_FNAME = _URL.split("/")[-1]


def _download_raw_counts(output_dir: Union[str, Path]) -> None:
    """Downloads sample RNASeq data (raw counts).

    Parameters
    ----------
    output_dir : Union[str, Path]
        Local directory where the sample data will be downloaded.
    """

    fname = str(Path(output_dir).joinpath(_RAW_COUNTS_FNAME))
    with urlopen(_URL) as response, open(fname, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    # Check that file was downloaded
    if not Path(fname).is_file():
        raise FileNotFoundError(f"Failed to download sample data from {_URL}!")


_download_raw_counts(Path.cwd())

df = pd.read_csv(Path.cwd().joinpath(_RAW_COUNTS_FNAME), sep="\t")

print(df.head())
