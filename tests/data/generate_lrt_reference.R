## Generate the R DESeq2 reference outputs used by the PyDESeq2 likelihood ratio
## test (LRT) suite.
##
## These files are the ground truth that `tests/test_pydeseq2.py` compares the
## PyDESeq2 LRT implementation against, mirroring how the existing Wald-test
## reference data was produced. They use DESeq2 defaults (parametric dispersion
## trend, median-of-ratios size factors, Cooks refitting), matching the defaults
## of `DeseqDataSet`/`DeseqStats`.
##
## Usage (from the repository root):
##     Rscript tests/data/generate_lrt_reference.R .
##
## Requires: DESeq2 (reference data was generated with DESeq2 1.46.0).

suppressMessages(library(DESeq2))

repo <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(repo)) repo <- "."

counts   <- read.csv(file.path(repo, "datasets/synthetic/test_counts.csv"),   row.names = 1)  # genes x samples
metadata <- read.csv(file.path(repo, "datasets/synthetic/test_metadata.csv"), row.names = 1)  # samples x factors

## Run an LRT and write both the default `results()` output (independent
## filtering ON) and a version with independent filtering OFF (which isolates the
## raw LRT statistic and p-value from the p-value adjustment step).
run_lrt <- function(counts, metadata, full, reduced, name, outfile) {
  dds <- DESeqDataSetFromMatrix(countData = counts, colData = metadata, design = full)
  dds <- DESeq(dds, test = "LRT", reduced = reduced, quiet = TRUE)
  write.csv(as.data.frame(results(dds, name = name)), outfile)
  write.csv(
    as.data.frame(results(dds, name = name, independentFiltering = FALSE)),
    sub("\\.csv$", "_no_independent_filtering.csv", outfile)
  )
  invisible(dds)
}

## --- Clean data (no outliers) -------------------------------------------------
meta <- metadata
meta$condition <- factor(meta$condition, levels = c("A", "B"))
meta$group     <- factor(meta$group,     levels = c("X", "Y"))

# Case A: single factor, full = ~condition, reduced = ~1 (df = 1)
run_lrt(counts, meta, ~condition, ~1, "condition_B_vs_A",
        file.path(repo, "tests/data/single_factor/r_test_res_lrt.csv"))

# Case B: multi factor, full = ~group + condition, reduced = ~group (df = 1)
run_lrt(counts, meta, ~group + condition, ~group, "condition_B_vs_A",
        file.path(repo, "tests/data/multi_factor/r_test_res_lrt.csv"))

# Case C: multi factor, full = ~group + condition, reduced = ~1 (df = 2)
run_lrt(counts, meta, ~group + condition, ~1, "condition_B_vs_A",
        file.path(repo, "tests/data/multi_factor/r_test_res_lrt_reduced_intercept.csv"))

## --- Data with Cooks outliers (triggers refitting) ---------------------------
## Same mild outlier injection as the existing Wald outlier test.
counts_out <- counts
counts_out["gene1", "sample1"]  <- 2000L
counts_out["gene7", "sample11"] <- 1000L

dds_out <- DESeqDataSetFromMatrix(countData = counts_out, colData = meta,
                                  design = ~group + condition)
dds_out <- DESeq(dds_out, test = "LRT", reduced = ~group, quiet = TRUE)
write.csv(as.data.frame(results(dds_out, name = "condition_B_vs_A")),
          file.path(repo, "tests/data/multi_factor/r_test_res_lrt_outliers.csv"))

cat("Done. n replaced (outlier case):", sum(mcols(dds_out)$replace, na.rm = TRUE), "\n")
