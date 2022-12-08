## Test data

This folder contains data for the pytest CI.

The `tests_clinical.csv` and `test_counts.csv` files were generated using the `makeExampleDESeqDataSet` 
of [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) (with parameters `m = 100`, `n = 10`,
and `betaSD = 1`).

The remainder of the files (`r_test_*`) correspond to the output of DESeq2 with default settings on this dataset, 
against which the output of PyDESeq2 is tested in the `test_pydeseq2.py` pytest when opening PRs.