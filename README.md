# Prediction of the Bacterial Resistance to Colistin antibiotic


## Project Overview
This repository contains a machine learning pipeline to predict Minimum Inhibitory Concentration (MIC) changes in *E. coli* populations under colistin selection pressure, using genomic data from 68 experimentally evolved bacterial genomes. The workflow links genetic variations (SNPs, indels, gene presence/absence) to phenotypic resistance evolution.

---

## Key Features
- **Inputs**: 
  - 68 evolved bacterial genomes (FASTQ/assemblies)
  - Ancestral reference genome (ATCC_14028.fna)
  - MIC measurements (log2-transformed)
  
- **Outputs**:
  - MIC prediction model (genotype → phenotype)
  - Ranked list of resistance-associated genetic features
  
 
  - # Bacterial Persistence & Colistin Resistance Prediction

## Project Overview
This repository contains a machine learning pipeline to predict Minimum Inhibitory Concentration (MIC) changes in *E. coli* populations under colistin selection pressure, using genomic data from 68 experimentally evolved bacterial genomes. The workflow links genetic variations (SNPs, indels, gene presence/absence) to phenotypic resistance evolution.

---

## Key Features
- **Inputs**: 
  - 68 evolved bacterial genomes (FASTQ/assemblies)
  - Ancestral reference genome (ATCC_14028.fna)
  - MIC measurements (log2-transformed)
  
- **Outputs**:
  - MIC prediction model (genotype → phenotype)
  - Ranked list of resistance-associated genetic features
  - Phylogeny-aware performance validation

---

## Quick Start

### Dependencies
```bash
# Create conda environment
conda create -n mic_pred -c bioconda -c conda-forge \
  snippy=4.6.0 \
  roary=3.13.0 \
  pyseer=1.3.7 \
  r-ggplot2=3.4.0 \
  python=3.10 \
  scikit-learn=1.2.2 \
  xgboost=1.7.3 \
  shap=0.41.0

conda activate mic_pred

├── data/
│   ├── raw/                  # FASTQ files (symlink or store externally)
│   ├── reference/            # ATCC_14028.fna, .gff
│   └── mic_measurements.csv  
├── results/
│   ├── core/                 # Snippy output (core genome alignment)
│   ├── pan_genome/           # Roary output
│   └── models/               # Saved ML models
├── notebooks/                # Exploratory analyses
└── scripts/                  # Pipeline workflows


This README provides a balance between technical reproducibility and biological interpretation, critical for translational antimicrobial resistance studies. Let me know if you need implementation support for any component!
