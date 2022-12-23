import os
from pathlib import Path

import numpy as np
import pandas as pd

import tests
from pydeseq2.DeseqDataSet import DeseqDataSet
from pydeseq2.DeseqStats import DeseqStats


def test_zero_genes(tol=0.02):
    """Test that genes with only 0 counts are handled by DeseqDataSet et DeseqStats,
    and that NaNs are returned
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    counts_df = pd.read_csv(
        os.path.join(test_path, "data/test_counts.csv"), index_col=0
    ).T
    clinical_df = pd.read_csv(
        os.path.join(test_path, "data/test_clinical.csv"), index_col=0
    )

    n, m = counts_df.shape

    # Introduce a few genes with fully 0 counts
    np.random.seed(42)
    zero_genes = counts_df.columns[np.random.choice(m, size=m // 3, replace=False)]
    counts_df[zero_genes] = 0

    # Run analysis
    dds = DeseqDataSet(counts_df, clinical_df, design_factor="condition")
    dds.deseq2()

    # check that the corresponding parameters are NaN
    assert dds.dispersions.loc[zero_genes].isna().all()
    assert dds.LFCs.loc[zero_genes].isna().all().all()

    res = DeseqStats(dds)
    res_df = res.summary()

    # check that the corresponding stats are NaN
    assert (res_df.loc[zero_genes].baseMean == 0).all()
    assert res_df.loc[zero_genes].log2FoldChange.isna().all()
    assert res_df.loc[zero_genes].lfcSE.isna().all()
    assert res_df.loc[zero_genes].stat.isna().all()
    assert res_df.loc[zero_genes].pvalue.isna().all()
    assert res_df.loc[zero_genes].padj.isna().all()
