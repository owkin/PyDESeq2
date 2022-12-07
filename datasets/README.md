# TCGA Dataset

Data from The Cancer Genome Atlas can be obtained [here](https://portal.gdc.cancer.gov/).

## Folder organisation

The `tcga_data` is supposed to be organized in the following way:

```
PyDESeq2
│
└───datasets
        │
        └───tcga_data   
            │   centers.csv
            │
            └───Clinical
            │       TCGA-BRCA_clinical.tsv.gz
            │       TCGA-COAD_clinical.tsv.gz
            │       TCGA-LUAD_clinical.tsv.gz
            │       TCGA-LUSC_clinical.tsv.gz
            │       TCGA-PAAD_clinical.tsv.gz
            │       TCGA-PRAD_clinical.tsv.gz
            │       TCGA-READ_clinical.tsv.gz
            │       TCGA-SKCM_clinical.tsv.gz
            │   
            └───Gene_expressions
                    TCGA-BRCA_raw_RNAseq.tsv.gz
                    TCGA-COAD_raw_RNAseq.tsv.gz
                    TCGA-LUAD_raw_RNAseq.tsv.gz
                    TCGA-LUSC_raw_RNAseq.tsv.gz
                    TCGA-PAAD_raw_RNAseq.tsv.gz
                    TCGA-PRAD_raw_RNAseq.tsv.gz
                    TCGA-READ_raw_RNAseq.tsv.gz
                    TCGA-SKCM_raw_RNAseq.tsv.gz
```
