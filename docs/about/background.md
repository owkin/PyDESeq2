# About PyDESeq2

[DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) {cite:p}`love2014moderated` is the reference method for differential expression analysis of bulk RNA-seq data.
It models raw counts with a negative binomial distribution, shares information across genes to stabilise the per-gene dispersion estimates, and tests log fold-changes with a Wald test.
It is written in R.

PyDESeq2 {cite:p}`muzellec2023pydeseq2` reimplements that method in Python, on {class}`~anndata.AnnData` objects, so that a differential expression step fits inside a Python analysis without leaving the language.

## Scope

Current features broadly correspond to the default settings of DESeq2 (v1.34.0):

- single-factor and multi-factor analysis, with categorical or continuous factors
- Wald tests, with Cooks filtering and independent filtering
- optional [apeGLM](https://academic.oup.com/bioinformatics/article/35/12/2084/5159452) log fold-change shrinkage {cite:p}`zhu2019heavy`
- variance-stabilising transformation

## Differences from DESeq2

PyDESeq2 is a reimplementation from scratch rather than a port, so retrieved values may differ slightly from those of DESeq2, and not every DESeq2 feature exists.
If a feature you rely on is missing, open an issue on [GitHub](https://github.com/scverse/PyDESeq2/issues).
