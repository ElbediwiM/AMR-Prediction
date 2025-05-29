#!/usr/bin/env python3
"""
E. coli SNP and MIC Analysis
This script performs exploratory data analysis and builds predictive models
to identify SNPs correlated with MIC values in E. coli genomes evolved with colistin.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the data
print("Loading data...")
# Skip the first line which contains '<MIC>' and read the actual header on the second line
mic_data = pd.read_csv('MIC.txt', sep='\t', skiprows=1)
snp_data = pd.read_excel('Snps.xlsx')

print(f"MIC data shape: {mic_data.shape}")
print(f"SNP data shape: {snp_data.shape}")

# Display the first few rows of each dataset
print("\nMIC data preview:")
print(mic_data.head())

print("\nSNP data preview:")
print(snp_data.head())

# Exploratory Data Analysis of MIC values
print("\nPerforming exploratory analysis of MIC values...")

# Summary statistics of MIC values
print("\nMIC Summary Statistics:")
mic_stats = mic_data['mic'].describe()
print(mic_stats)

# Plot MIC distribution
plt.figure(figsize=(10, 6))
sns.histplot(mic_data['mic'], kde=True, bins=10)
plt.title('Distribution of MIC Values', fontsize=16)
plt.xlabel('MIC Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig('figures/mic_distribution.png', dpi=300, bbox_inches='tight')

# Plot MIC values as boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(y=mic_data['mic'])
plt.title('Boxplot of MIC Values', fontsize=16)
plt.ylabel('MIC Value', fontsize=14)
plt.savefig('figures/mic_boxplot.png', dpi=300, bbox_inches='tight')

# Exploratory Analysis of SNP data
print("\nPerforming exploratory analysis of SNP data...")

# Count SNPs by chromosome
plt.figure(figsize=(12, 6))
sns.countplot(y='CHROM', data=snp_data, order=snp_data['CHROM'].value_counts().index)
plt.title('SNP Count by Chromosome', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Chromosome', fontsize=14)
plt.savefig('figures/snp_by_chromosome.png', dpi=300, bbox_inches='tight')

# Count SNPs by impact - convert to string first
snp_data['ANN[0].IMPACT'] = snp_data['ANN[0].IMPACT'].astype(str)
plt.figure(figsize=(12, 6))
sns.countplot(y='ANN[0].IMPACT', data=snp_data, order=snp_data['ANN[0].IMPACT'].value_counts().index)
plt.title('SNP Count by Impact', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Impact', fontsize=14)
plt.savefig('figures/snp_by_impact.png', dpi=300, bbox_inches='tight')

# Count SNPs by effect - convert to string first
snp_data['ANN[0].EFFECT'] = snp_data['ANN[0].EFFECT'].astype(str)
plt.figure(figsize=(12, 8))
effect_counts = snp_data['ANN[0].EFFECT'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=effect_counts.values, y=effect_counts.index)
plt.title('SNP Count by Effect', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Effect', fontsize=14)
plt.savefig('figures/snp_by_effect.png', dpi=300, bbox_inches='tight')

# Count SNPs by gene (top 20) - convert to string first
snp_data['ANN[0].GENE'] = snp_data['ANN[0].GENE'].astype(str)
plt.figure(figsize=(14, 10))
gene_counts = snp_data['ANN[0].GENE'].value_counts().head(20)
sns.barplot(x=gene_counts.values, y=gene_counts.index)
plt.title('Top 20 Genes by SNP Count', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Gene', fontsize=14)
plt.savefig('figures/top_genes_by_snp_count.png', dpi=300, bbox_inches='tight')

# Save summary statistics to file
with open('results/data_summary.txt', 'w') as f:
    f.write("MIC Data Summary:\n")
    f.write(f"Number of samples: {mic_data.shape[0]}\n")
    f.write(f"MIC value range: {mic_data['mic'].min()} - {mic_data['mic'].max()}\n")
    f.write(f"MIC value mean: {mic_data['mic'].mean():.2f}\n")
    f.write(f"MIC value median: {mic_data['mic'].median()}\n\n")
    
    f.write("SNP Data Summary:\n")
    f.write(f"Total SNPs: {snp_data.shape[0]}\n")
    f.write(f"Number of chromosomes: {snp_data['CHROM'].nunique()}\n")
    f.write(f"Number of genes with SNPs: {snp_data['ANN[0].GENE'].nunique()}\n")
    f.write(f"Impact categories: {', '.join(snp_data['ANN[0].IMPACT'].unique())}\n")
    f.write(f"Most common effect: {snp_data['ANN[0].EFFECT'].value_counts().index[0]}\n")
    f.write(f"Most affected gene: {snp_data['ANN[0].GENE'].value_counts().index[0]} ({snp_data['ANN[0].GENE'].value_counts().values[0]} SNPs)\n")

print("\nExploratory analysis completed. Results saved to 'figures/' and 'results/' directories.")
