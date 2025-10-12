## Test data

This folder contains data for the pytest CI.

The files in `single_factor` and `multi_factor` contain the outputs of DESeq2 (v1.34.0) on the synthetic data provided
in `/datasets/synthetic/`, respectively using `~condition` and `~condition + group` as design. More precisely:

- `r_iterative_size_factors.csv` contains DESeq2's `estimateSizeFactorsIterate` output,
- `r_lfc_shrink.csv` contains DESeq2 results after running `lfcShrink`,
- `r_test_dispersions.csv` contains DESeq2 dispersions estimates (post-filtering and refitting),
- `r_test_res.csv` contains DESeq2's `results` output,
- `r_test_size_factors.csv` contains DESeq2's `estimateSizeFactors` output,
- `r_vst.csv` contains DESeq2's `varianceStabilizingTransformation` output with `blind=TRUE` and `fitType="parametric"`,
- `r_vst_with_design.csv` contains DESeq2's `varianceStabilizingTransformation` output with `blind=FALSE` and `fitType="parametric"`.
