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

### Likelihood ratio test (LRT) reference data

The `r_test_res_lrt*.csv` files contain the output of DESeq2's `results()` after running
`DESeq(dds, test="LRT", reduced=...)`, and are used by the LRT tests. They were generated
with DESeq2 1.46.0 by `generate_lrt_reference.R` (run `Rscript tests/data/generate_lrt_reference.R .`
from the repository root to regenerate them). For each case, a `*_no_independent_filtering.csv`
variant is also stored (i.e. `results(..., independentFiltering=FALSE)`), which isolates the raw
LRT statistic and p-value from the p-value adjustment step.

- `single_factor/r_test_res_lrt.csv`: full `~condition` vs reduced `~1` (`df=1`),
- `multi_factor/r_test_res_lrt.csv`: full `~group + condition` vs reduced `~group` (`df=1`),
- `multi_factor/r_test_res_lrt_reduced_intercept.csv`: full `~group + condition` vs reduced `~1` (`df=2`),
- `multi_factor/r_test_res_lrt_outliers.csv`: same as above (`~group + condition` vs `~group`) but with
  injected Cooks outliers that trigger outlier replacement and model refitting.
