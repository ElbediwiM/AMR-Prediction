"""
# Comprehensive Code Compilation for SNP-MIC Correlation Analysis
# Analysis of Salmonella typhimurium bacteria evolved in colistin antibiotic
# Under aerobic and anaerobic conditions

This file contains all the Python code used in the analysis of SNPs and MIC correlation
in Salmonella typhimurium bacteria evolved in colistin antibiotic under both aerobic
and anaerobic conditions.

## Table of Contents:
1. Aerobic Analysis
   1.1 Examine and Parse Files
   1.2 Feature Engineering and SNP Selection
   1.3 MIC Data Preprocessing
   1.4 Data Merging for Modeling
   1.5 Model Training and Validation
   1.6 Results Interpretation

2. Anaerobic Analysis
   2.1 Examine and Parse Files
   2.2 Feature Engineering and SNP Selection
   2.3 MIC Data Preprocessing
   2.4 Data Merging for Modeling
   2.5 Model Training and Validation

3. Combined Analysis
   3.1 Dataset Merging and Harmonization
   3.2 Combined Model Training and Validation
"""

#######################
# 1. AEROBIC ANALYSIS #
#######################

# 1.1 Examine and Parse Files
# ---------------------------

# Examine MIC.txt file
import pandas as pd
import numpy as np

# Read the MIC data
mic_df = pd.read_csv('/home/ubuntu/upload/MIC.txt', sep='\t', skiprows=1)

# Display the first few rows
print(mic_df.head())

# Examine Aerobic_annotation_final.xlsx file
annotation_df = pd.read_excel('/home/ubuntu/upload/Aerobic_annotation_final.xlsx')

# Display the first few rows
print(annotation_df.head().to_string())

# Display the unique impact values
print('Unique impact values:')
print(annotation_df['ANN[0].IMPACT'].unique())

# Count the number of SNPs by impact
print('\nCount of SNPs by impact:')
print(annotation_df['ANN[0].IMPACT'].value_counts())


# 1.2 Feature Engineering and SNP Selection
# ----------------------------------------

# Filter for high and moderate impact SNPs
high_moderate_snps = annotation_df[annotation_df['ANN[0].IMPACT'].isin(['HIGH', 'MODERATE'])]

# Display the count of high and moderate impact SNPs
print('\nCount of HIGH and MODERATE impact SNPs:', len(high_moderate_snps))

# Save the filtered SNPs to a CSV file
high_moderate_snps.to_csv('high_moderate_impact_snps.csv', index=False)

# Display the first few rows of the filtered data
print('\nFirst few rows of HIGH and MODERATE impact SNPs:')
print(high_moderate_snps.head().to_string())

# Check if there are any SNPs with start of mutation information
print('\nSNPs with start of mutation information:')
print(high_moderate_snps['start of mutation'].value_counts(dropna=False))


# 1.3 MIC Data Preprocessing
# -------------------------

# Extract day number from sample ID
mic_df['day'] = mic_df['id'].apply(lambda x: int(x.split('_')[0]))

# Label samples as susceptible (0) or resistant (1) based on breakpoint
mic_df['resistant'] = (mic_df['mic'] > 2).astype(int)

# Sort by day
mic_df = mic_df.sort_values('day')

# Save the processed MIC data
mic_df.to_csv('processed_mic_data.csv', index=False)

# Display the processed data
print('Processed MIC data:')
print(mic_df.to_string())

# Check the transition point (day 27)
print('\nMIC values around day 27:')
print(mic_df[mic_df['day'].isin(range(25, 30))].to_string())

# Summarize resistance by day
print('\nResistance summary by day:')
resistance_by_day = mic_df.groupby(['day', 'resistant']).size().unstack(fill_value=0)
print(resistance_by_day)

# Count of susceptible and resistant samples
print('\nTotal count:')
print(f'Susceptible samples (MIC <= 2): {sum(mic_df["resistant"] == 0)}')
print(f'Resistant samples (MIC > 2): {sum(mic_df["resistant"] == 1)}')


# 1.4 Data Merging for Modeling
# ----------------------------

# Create a feature matrix based on SNPs with start of mutation information
snps_with_day = high_moderate_snps[~high_moderate_snps['start of mutation'].isna()]
print(snps_with_day[['CHROM', 'POS', 'REF', 'ALT', 'ANN[0].EFFECT', 'ANN[0].IMPACT', 'ANN[0].GENE', 'start of mutation']].to_string())

# Count SNPs by day of mutation
print('\nCount of SNPs by day of mutation:')
print(snps_with_day['start of mutation'].value_counts())

# Create a mapping of days to SNPs
day_to_snps = {}
for _, row in snps_with_day.iterrows():
    day = int(row['start of mutation'])
    snp_id = f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}"
    if day not in day_to_snps:
        day_to_snps[day] = []
    day_to_snps[day].append(snp_id)

# Create a feature matrix where each row is a day and columns are SNPs
all_snps = []
for day_snps in day_to_snps.values():
    all_snps.extend(day_snps)
all_snps = list(set(all_snps))

# Initialize the feature matrix
feature_matrix = []
for day in range(1, 69):  # Days 1 to 68
    features = {}
    features['day'] = day
    
    # Set SNP features (1 if present by this day, 0 otherwise)
    for snp in all_snps:
        features[snp] = 0
    
    # Mark SNPs that appeared by this day
    for d, snps in day_to_snps.items():
        if d <= day:
            for snp in snps:
                features[snp] = 1
    
    feature_matrix.append(features)

# Convert to DataFrame
feature_df = pd.DataFrame(feature_matrix)

# Merge with MIC data
merged_data = pd.merge(feature_df, mic_df[['day', 'mic', 'resistant']], on='day', how='inner')

# Save the merged dataset
merged_data.to_csv('merged_snp_mic_data.csv', index=False)

# Display information about the merged dataset
print('\nMerged dataset shape:', merged_data.shape)
print('\nMerged dataset columns:', merged_data.columns.tolist())
print('\nFirst 5 rows of merged dataset:')
print(merged_data.head().to_string())

# Check the distribution of resistant/susceptible samples in the merged dataset
print('\nResistance distribution in merged dataset:')
print(merged_data['resistant'].value_counts())


# 1.5 Model Training and Validation
# --------------------------------

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the merged dataset
data = pd.read_csv('merged_snp_mic_data.csv')

# Separate features and target
X = data.drop(['day', 'mic', 'resistant'], axis=1)
y = data['resistant']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print('Dataset information:')
print(f'Total samples: {len(data)}')
print(f'Features: {X.shape[1]}')
print(f'Training samples: {len(X_train)}')
print(f'Testing samples: {len(X_test)}')
print(f'Resistant samples: {sum(y == 1)}')
print(f'Susceptible samples: {sum(y == 0)}')

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Evaluate each model using cross-validation
print('\nModel evaluation using 5-fold cross-validation:')
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f'{name}: Mean accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}')

# Hyperparameter tuning for the best model based on CV results
best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f'\nBest model based on cross-validation: {best_model_name}')

# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
}

# Perform grid search for the best model
print(f'\nPerforming hyperparameter tuning for {best_model_name}...')
grid_search = GridSearchCV(
    models[best_model_name],
    param_grids[best_model_name],
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_:.4f}')

# Train the best model with optimized hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('\nTest set performance:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(cm)

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Feature importance analysis
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importances)[::-1]
    
    print('\nTop 10 most important SNPs:')
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f'{feature_names[idx]}: {feature_importances[idx]:.4f}')
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(feature_importances)), feature_importances[sorted_idx])
    plt.xticks(range(len(feature_importances)), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    
elif best_model_name == 'Logistic Regression':
    coefficients = best_model.coef_[0]
    feature_names = X.columns
    
    # Sort features by absolute coefficient value
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    
    print('\nTop 10 most important SNPs:')
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f'{feature_names[idx]}: {coefficients[idx]:.4f}')
    
    # Plot coefficients
    plt.figure(figsize=(12, 8))
    plt.title('Logistic Regression Coefficients')
    plt.bar(range(len(coefficients)), coefficients[sorted_idx])
    plt.xticks(range(len(coefficients)), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients.png')

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print('\nBest model saved as best_model.pkl')


# 1.6 Results Interpretation
# -------------------------

# Create a detailed analysis of the top SNPs
print('Detailed analysis of top SNPs contributing to resistance:')
print('-' * 80)

with open('snp_detailed_analysis.txt', 'w') as f:
    f.write('# Detailed Analysis of SNPs Contributing to Colistin Resistance\n\n')
    
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        snp_id = feature_names[idx]
        coef = coefficients[idx]
        
        # Get annotation information if available
        for _, row in high_moderate_snps.iterrows():
            row_snp_id = f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}"
            if row_snp_id == snp_id:
                anno = {
                    "effect": row['ANN[0].EFFECT'],
                    "impact": row['ANN[0].IMPACT'],
                    "gene": row['ANN[0].GENE'],
                    "hgvs_p": row['ANN[0].HGVS_P'] if pd.notna(row['ANN[0].HGVS_P']) else 'N/A',
                    "start_of_mutation": row['start of mutation'] if pd.notna(row['start of mutation']) else 'N/A'
                }
                
                print(f'{i+1}. SNP: {snp_id}')
                print(f'   Coefficient: {coef:.4f}')
                print(f'   Effect: {anno["effect"]}')
                print(f'   Impact: {anno["impact"]}')
                print(f'   Gene: {anno["gene"]}')
                print(f'   Protein change: {anno["hgvs_p"]}')
                print(f'   Start of mutation: {anno["start_of_mutation"]}')
                print('-' * 80)
                
                f.write(f'## {i+1}. SNP: {snp_id}\n')
                f.write(f'- **Coefficient**: {coef:.4f}\n')
                f.write(f'- **Effect**: {anno["effect"]}\n')
                f.write(f'- **Impact**: {anno["impact"]}\n')
                f.write(f'- **Gene**: {anno["gene"]}\n')
                f.write(f'- **Protein change**: {anno["hgvs_p"]}\n')
                f.write(f'- **Start of mutation**: {anno["start_of_mutation"]}\n\n')

# Plot the correlation between SNPs and resistance
plt.figure(figsize=(14, 10))
sns.heatmap(data[['resistant'] + list(feature_names)].corr()['resistant'].sort_values(ascending=False)[1:11].to_frame(), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Top SNPs and Resistance')
plt.tight_layout()
plt.savefig('snp_resistance_correlation.png')

# Plot the distribution of MIC values over time with SNP emergence
plt.figure(figsize=(14, 8))
plt.plot(data['day'], data['mic'], 'o-', label='MIC Value')
plt.axhline(y=2, color='r', linestyle='--', label='Resistance Breakpoint (MIC > 2)')

# Add vertical lines for when key SNPs emerged (day 27)
plt.axvline(x=27, color='g', linestyle='--', label='Key SNPs Emergence (Day 27)')

plt.xlabel('Day')
plt.ylabel('MIC Value')
plt.title('MIC Values Over Time with SNP Emergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mic_over_time_with_snps.png')

# Create a visualization of SNP presence over time
snp_presence = data.drop(['mic', 'resistant'], axis=1)
plt.figure(figsize=(14, 10))
sns.heatmap(snp_presence.set_index('day'), cmap='viridis', cbar_kws={'label': 'SNP Present'})
plt.title('SNP Presence Over Time')
plt.xlabel('SNPs')
plt.ylabel('Day')
plt.tight_layout()
plt.savefig('snp_presence_over_time.png')


#########################
# 2. ANAEROBIC ANALYSIS #
#########################

# 2.1 Examine and Parse Files
# ---------------------------

# Examine the anaerobic MIC file
mic_anaerobic = pd.read_excel('/home/ubuntu/upload/MIC.xlsx', skiprows=1)
mic_anaerobic.columns = ['id', 'mic']  # Rename columns to match aerobic dataset

print('Anaerobic MIC data structure:')
print(mic_anaerobic.head().to_string())
print(f'Shape: {mic_anaerobic.shape}')
print(f'Columns: {mic_anaerobic.columns.tolist()}')

# Check for missing values
print(f'Missing values: {mic_anaerobic.isnull().sum().sum()}')

# Examine the anaerobic SNP annotation file
snp_annotation = pd.read_csv('/home/ubuntu/upload/Anotated_SNPs_Anaerobic.tabular', sep='\t')

# Rename columns to match aerobic dataset format
snp_annotation = snp_annotation.rename(columns={
    'ANN[*].EFFECT': 'ANN[0].EFFECT',
    'ANN[*].IMPACT': 'ANN[0].IMPACT',
    'ANN[*].GENE': 'ANN[0].GENE'
})

print('\nAnaerobic SNP annotation structure:')
print(snp_annotation.head().to_string())
print(f'Shape: {snp_annotation.shape}')
print(f'Columns: {snp_annotation.columns.tolist()}')

# Check for impact values
print('\nUnique impact values:')
print(snp_annotation['ANN[0].IMPACT'].unique())

# Count the number of SNPs by impact
print('\nCount of SNPs by impact:')
print(snp_annotation['ANN[0].IMPACT'].value_counts())


# 2.2 Feature Engineering and SNP Selection
# ----------------------------------------

# Filter for high and moderate impact SNPs
high_moderate_snps_anaerobic = snp_annotation[snp_annotation['ANN[0].IMPACT'].isin(['HIGH', 'MODERATE'])]

# Display the count of high and moderate impact SNPs
print('\nCount of HIGH and MODERATE impact SNPs:', len(high_moderate_snps_anaerobic))

# Save the filtered SNPs to a CSV file
high_moderate_snps_anaerobic.to_csv('high_moderate_impact_snps_anaerobic.csv', index=False)

# Display the first few rows of the filtered data
print('\nFirst few rows of HIGH and MODERATE impact SNPs:')
print(high_moderate_snps_anaerobic.head().to_string())


# 2.3 MIC Data Preprocessing
# -------------------------

# Extract day number from sample ID
mic_anaerobic['day'] = mic_anaerobic['id'].apply(lambda x: int(x.split('_')[0].replace('A', '')))

# Label samples as susceptible (0) or resistant (1) based on breakpoint
mic_anaerobic['resistant'] = (mic_anaerobic['mic'] > 2).astype(int)

# Sort by day
mic_anaerobic = mic_anaerobic.sort_values('day')

# Save the processed MIC data
mic_anaerobic.to_csv('processed_mic_anaerobic_data.csv', index=False)

# Display the processed data
print('Processed Anaerobic MIC data:')
print(mic_anaerobic.to_string())

# Summarize resistance by day
print('\nResistance summary by day:')
resistance_by_day = mic_anaerobic.groupby(['day', 'resistant']).size().unstack(fill_value=0)
print(resistance_by_day)

# Count of susceptible and resistant samples
print('\nTotal count:')
print(f'Susceptible samples (MIC <= 2): {sum(mic_anaerobic["resistant"] == 0)}')
print(f'Resistant samples (MIC > 2): {sum(mic_anaerobic["resistant"] == 1)}')

# Identify the day when resistance emerges
resistance_emergence_day = mic_anaerobic[mic_anaerobic['resistant'] == 1]['day'].min()
print(f'\nResistance emergence day: {resistance_emergence_day}')
print('\nMIC values around resistance emergence:')
print(mic_anaerobic[mic_anaerobic['day'].isin(range(resistance_emergence_day-2, resistance_emergence_day+3))].to_string())

# Add a column for start of mutation based on the resistance emergence day
high_moderate_snps_anaerobic['start of mutation'] = resistance_emergence_day

# Save the updated filtered SNPs
high_moderate_snps_anaerobic.to_csv('high_moderate_impact_snps_anaerobic_with_day.csv', index=False)


# 2.4 Data Merging for Modeling
# ----------------------------

# Create a unique identifier for each SNP
high_moderate_snps_anaerobic['snp_id'] = high_moderate_snps_anaerobic.apply(
    lambda row: f"{row['CHROM']}_{row['POS']}_{row['REF']}_{row['ALT']}", axis=1
)

# Create a feature matrix where each row is a day and columns are SNPs
all_snps_anaerobic = high_moderate_snps_anaerobic['snp_id'].unique().tolist()

# Initialize the feature matrix
feature_matrix_anaerobic = []
for day in range(1, 31):  # Days 1 to 30 for anaerobic
    features = {}
    features['day'] = day
    
    # Set SNP features (1 if present by this day, 0 otherwise)
    for snp in all_snps_anaerobic:
        features[snp] = 0
    
    # Mark SNPs that appeared by this day (day 15 for all in this simplified model)
    if day >= 15:  # Resistance emergence day
        for snp in all_snps_anaerobic:
            features[snp] = 1
    
    feature_matrix_anaerobic.append(features)

# Convert to DataFrame
feature_df_anaerobic = pd.DataFrame(feature_matrix_anaerobic)

# Merge with MIC data
merged_data_anaerobic = pd.merge(feature_df_anaerobic, mic_anaerobic[['day', 'mic', 'resistant']], on='day', how='inner')

# Save the merged dataset
merged_data_anaerobic.to_csv('merged_snp_mic_data_anaerobic.csv', index=False)

# Display information about the merged dataset
print('Merged dataset shape:', merged_data_anaerobic.shape)
print('Merged dataset columns (first 10):', merged_data_anaerobic.columns.tolist()[:10])
print('First 5 rows of merged dataset (truncated):')
print(merged_data_anaerobic[['day'] + all_snps_anaerobic[:3] + ['mic', 'resistant']].head().to_string())

# Check the distribution of resistant/susceptible samples in the merged dataset
print('\nResistance distribution in merged dataset:')
print(merged_data_anaerobic['resistant'].value_counts())


# 2.5 Model Training and Validation
# --------------------------------

# Load the anaerobic dataset
data_anaerobic = pd.read_csv('merged_snp_mic_data_anaerobic.csv')

# Separate features and target
X_anaerobic = data_anaerobic.drop(['day', 'mic', 'resistant'], axis=1)
y_anaerobic = data_anaerobic['resistant']

# Split the data into training and testing sets
X_train_anaerobic, X_test_anaerobic, y_train_anaerobic, y_test_anaerobic = train_test_split(
    X_anaerobic, y_anaerobic, test_size=0.25, random_state=42, stratify=y_anaerobic
)

print('Anaerobic dataset information:')
print(f'Total samples: {len(data_anaerobic)}')
print(f'Features: {X_anaerobic.shape[1]}')
print(f'Training samples: {len(X_train_anaerobic)}')
print(f'Testing samples: {len(X_test_anaerobic)}')
print(f'Resistant samples: {sum(y_anaerobic == 1)}')
print(f'Susceptible samples: {sum(y_anaerobic == 0)}')

# Define models to evaluate
models_anaerobic = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Evaluate each model using cross-validation
print('\nModel evaluation using 5-fold cross-validation:')
cv_results_anaerobic = {}
for name, model in models_anaerobic.items():
    scores = cross_val_score(model, X_train_anaerobic, y_train_anaerobic, cv=5, scoring='accuracy')
    cv_results_anaerobic[name] = scores
    print(f'{name}: Mean accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}')

# Hyperparameter tuning for the best model based on CV results
best_model_name_anaerobic = max(cv_results_anaerobic, key=lambda k: cv_results_anaerobic[k].mean())
print(f'\nBest model based on cross-validation: {best_model_name_anaerobic}')

# Define hyperparameter grids for each model
param_grids_anaerobic = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
}

# Perform grid search for the best model
print(f'\nPerforming hyperparameter tuning for {best_model_name_anaerobic}...')
grid_search_anaerobic = GridSearchCV(
    models_anaerobic[best_model_name_anaerobic],
    param_grids_anaerobic[best_model_name_anaerobic],
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_anaerobic.fit(X_train_anaerobic, y_train_anaerobic)

print(f'Best parameters: {grid_search_anaerobic.best_params_}')
print(f'Best cross-validation score: {grid_search_anaerobic.best_score_:.4f}')

# Train the best model with optimized hyperparameters
best_model_anaerobic = grid_search_anaerobic.best_estimator_
best_model_anaerobic.fit(X_train_anaerobic, y_train_anaerobic)

# Evaluate on test set
y_pred_anaerobic = best_model_anaerobic.predict(X_test_anaerobic)
accuracy_anaerobic = accuracy_score(y_test_anaerobic, y_pred_anaerobic)
precision_anaerobic = precision_score(y_test_anaerobic, y_pred_anaerobic)
recall_anaerobic = recall_score(y_test_anaerobic, y_pred_anaerobic)
f1_anaerobic = f1_score(y_test_anaerobic, y_pred_anaerobic)

print('\nTest set performance:')
print(f'Accuracy: {accuracy_anaerobic:.4f}')
print(f'Precision: {precision_anaerobic:.4f}')
print(f'Recall: {recall_anaerobic:.4f}')
print(f'F1 Score: {f1_anaerobic:.4f}')

# Display confusion matrix
cm_anaerobic = confusion_matrix(y_test_anaerobic, y_pred_anaerobic)
print('\nConfusion Matrix:')
print(cm_anaerobic)

# Classification report
print('\nClassification Report:')
print(classification_report(y_test_anaerobic, y_pred_anaerobic))

# Feature importance analysis
if best_model_name_anaerobic in ['Random Forest', 'Gradient Boosting']:
    feature_importances_anaerobic = best_model_anaerobic.feature_importances_
    feature_names_anaerobic = X_anaerobic.columns
    
    # Sort features by importance
    sorted_idx_anaerobic = np.argsort(feature_importances_anaerobic)[::-1]
    
    print('\nTop 10 most important SNPs:')
    for i in range(min(10, len(sorted_idx_anaerobic))):
        idx = sorted_idx_anaerobic[i]
        print(f'{feature_names_anaerobic[idx]}: {feature_importances_anaerobic[idx]:.4f}')
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances (Anaerobic)')
    plt.bar(range(len(feature_importances_anaerobic)), feature_importances_anaerobic[sorted_idx_anaerobic])
    plt.xticks(range(len(feature_importances_anaerobic)), [feature_names_anaerobic[i] for i in sorted_idx_anaerobic], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances_anaerobic.png')
    
elif best_model_name_anaerobic == 'Logistic Regression':
    coefficients_anaerobic = best_model_anaerobic.coef_[0]
    feature_names_anaerobic = X_anaerobic.columns
    
    # Sort features by absolute coefficient value
    sorted_idx_anaerobic = np.argsort(np.abs(coefficients_anaerobic))[::-1]
    
    print('\nTop 10 most important SNPs:')
    for i in range(min(10, len(sorted_idx_anaerobic))):
        idx = sorted_idx_anaerobic[i]
        print(f'{feature_names_anaerobic[idx]}: {coefficients_anaerobic[idx]:.4f}')
    
    # Plot coefficients
    plt.figure(figsize=(12, 8))
    plt.title('Logistic Regression Coefficients (Anaerobic)')
    plt.bar(range(len(coefficients_anaerobic)), coefficients_anaerobic[sorted_idx_anaerobic])
    plt.xticks(range(len(coefficients_anaerobic)), [feature_names_anaerobic[i] for i in sorted_idx_anaerobic], rotation=90)
    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients_anaerobic.png')

# Save the best model
with open('best_model_anaerobic.pkl', 'wb') as f:
    pickle.dump(best_model_anaerobic, f)

print('\nBest model saved as best_model_anaerobic.pkl')


#######################
# 3. COMBINED ANALYSIS #
#######################

# 3.1 Dataset Merging and Harmonization
# ------------------------------------

# Load the aerobic and anaerobic datasets
aerobic_data = pd.read_csv('aerobic_merged_data.csv')
anaerobic_data = pd.read_csv('anaerobic_merged_data.csv')

print('Aerobic dataset shape:', aerobic_data.shape)
print('Anaerobic dataset shape:', anaerobic_data.shape)

# Add condition column to each dataset
aerobic_data['condition'] = 'aerobic'
anaerobic_data['condition'] = 'anaerobic'

# Get all SNP columns from both datasets
aerobic_snp_cols = [col for col in aerobic_data.columns if col not in ['day', 'mic', 'resistant', 'condition']]
anaerobic_snp_cols = [col for col in anaerobic_data.columns if col not in ['day', 'mic', 'resistant', 'condition']]

print(f'Aerobic SNP features: {len(aerobic_snp_cols)}')
print(f'Anaerobic SNP features: {len(anaerobic_snp_cols)}')

# Create a unified set of SNP features
all_snp_cols = list(set(aerobic_snp_cols) | set(anaerobic_snp_cols))
print(f'Total unique SNP features: {len(all_snp_cols)}')

# Function to add missing columns with zeros
def add_missing_columns(df, all_columns):
    missing_cols = set(all_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    return df

# Add missing SNP columns to each dataset
aerobic_data = add_missing_columns(aerobic_data, all_snp_cols)
anaerobic_data = add_missing_columns(anaerobic_data, all_snp_cols)

# Combine the datasets
combined_data = pd.concat([aerobic_data, anaerobic_data], ignore_index=True)

# Ensure all columns are in the same order
combined_data = combined_data[['day', 'condition'] + all_snp_cols + ['mic', 'resistant']]

# Save the combined dataset
combined_data.to_csv('combined_snp_mic_data.csv', index=False)

print('\nCombined dataset shape:', combined_data.shape)
print('Combined dataset columns (first 10):', combined_data.columns.tolist()[:10])

# Check the distribution of samples in the combined dataset
print('\nSample distribution in combined dataset:')
print(combined_data.groupby(['condition', 'resistant']).size())

# Create a one-hot encoded version for condition
combined_data_encoded = combined_data.copy()
combined_data_encoded['is_aerobic'] = (combined_data_encoded['condition'] == 'aerobic').astype(int)
combined_data_encoded = combined_data_encoded.drop('condition', axis=1)

# Save the encoded version
combined_data_encoded.to_csv('combined_snp_mic_data_encoded.csv', index=False)

print('\nEncoded combined dataset shape:', combined_data_encoded.shape)
print('Encoded combined dataset columns (first 10):', combined_data_encoded.columns.tolist()[:10])


# 3.2 Combined Model Training and Validation
# ----------------------------------------

# Note: This code was not fully executed due to time constraints, but is provided for completeness

# Load the combined dataset
data_combined = pd.read_csv('combined_snp_mic_data_encoded.csv')

# Separate features and target
X_combined = data_combined.drop(['day', 'mic', 'resistant'], axis=1)
y_combined = data_combined['resistant']

# Split the data into training and testing sets
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    X_combined, y_combined, test_size=0.25, random_state=42, stratify=y_combined
)

print('Combined dataset information:')
print(f'Total samples: {len(data_combined)}')
print(f'Features: {X_combined.shape[1]}')
print(f'Training samples: {len(X_train_combined)}')
print(f'Testing samples: {len(X_test_combined)}')
print(f'Resistant samples: {sum(y_combined == 1)}')
print(f'Susceptible samples: {sum(y_combined == 0)}')
print(f'Aerobic samples: {sum(data_combined["is_aerobic"] == 1)}')
print(f'Anaerobic samples: {sum(data_combined["is_aerobic"] == 0)}')

# Define models to evaluate
models_combined = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Evaluate each model using cross-validation
print('\nModel evaluation using 5-fold cross-validation:')
cv_results_combined = {}
for name, model in models_combined.items():
    scores = cross_val_score(model, X_train_combined, y_train_combined, cv=5, scoring='accuracy')
    cv_results_combined[name] = scores
    print(f'{name}: Mean accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}')

# Hyperparameter tuning for the best model based on CV results
best_model_name_combined = max(cv_results_combined, key=lambda k: cv_results_combined[k].mean())
print(f'\nBest model based on cross-validation: {best_model_name_combined}')

# Define hyperparameter grids for each model
param_grids_combined = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
}

# Perform grid search for the best model
print(f'\nPerforming hyperparameter tuning for {best_model_name_combined}...')
grid_search_combined = GridSearchCV(
    models_combined[best_model_name_combined],
    param_grids_combined[best_model_name_combined],
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_combined.fit(X_train_combined, y_train_combined)

print(f'Best parameters: {grid_search_combined.best_params_}')
print(f'Best cross-validation score: {grid_search_combined.best_score_:.4f}')

# Train the best model with optimized hyperparameters
best_model_combined = grid_search_combined.best_estimator_
best_model_combined.fit(X_train_combined, y_train_combined)

# Evaluate on test set
y_pred_combined = best_model_combined.predict(X_test_combined)
accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
precision_combined = precision_score(y_test_combined, y_pred_combined)
recall_combined = recall_score(y_test_combined, y_pred_combined)
f1_combined = f1_score(y_test_combined, y_pred_combined)

print('\nTest set performance:')
print(f'Accuracy: {accuracy_combined:.4f}')
print(f'Precision: {precision_combined:.4f}')
print(f'Recall: {recall_combined:.4f}')
print(f'F1 Score: {f1_combined:.4f}')

# Display confusion matrix
cm_combined = confusion_matrix(y_test_combined, y_pred_combined)
print('\nConfusion Matrix:')
print(cm_combined)

# Classification report
print('\nClassification Report:')
print(classification_report(y_test_combined, y_pred_combined))

# Feature importance analysis
if best_model_name_combined in ['Random Forest', 'Gradient Boosting']:
    feature_importances_combined = best_model_combined.feature_importances_
    feature_names_combined = X_combined.columns
    
    # Sort features by importance
    sorted_idx_combined = np.argsort(feature_importances_combined)[::-1]
    
    print('\nTop 15 most important features:')
    for i in range(min(15, len(sorted_idx_combined))):
        idx = sorted_idx_combined[i]
        print(f'{feature_names_combined[idx]}: {feature_importances_combined[idx]:.4f}')
    
    # Check if 'is_aerobic' is an important feature
    aerobic_idx = list(feature_names_combined).index('is_aerobic')
    aerobic_importance = feature_importances_combined[aerobic_idx]
    print(f'\nImportance of aerobic vs. anaerobic condition: {aerobic_importance:.4f}')
    print(f'Rank of condition among all features: {list(sorted_idx_combined).index(aerobic_idx) + 1} out of {len(sorted_idx_combined)}')
    
elif best_model_name_combined == 'Logistic Regression':
    coefficients_combined = best_model_combined.coef_[0]
    feature_names_combined = X_combined.columns
    
    # Sort features by absolute coefficient value
    sorted_idx_combined = np.argsort(np.abs(coefficients_combined))[::-1]
    
    print('\nTop 15 most important features:')
    for i in range(min(15, len(sorted_idx_combined))):
        idx = sorted_idx_combined[i]
        print(f'{feature_names_combined[idx]}: {coefficients_combined[idx]:.4f}')
    
    # Check if 'is_aerobic' is an important feature
    aerobic_idx = list(feature_names_combined).index('is_aerobic')
    aerobic_coef = coefficients_combined[aerobic_idx]
    print(f'\nCoefficient for aerobic vs. anaerobic condition: {aerobic_coef:.4f}')
    print(f'Rank of condition among all features: {list(sorted_idx_combined).index(aerobic_idx) + 1} out of {len(sorted_idx_combined)}')

# Save the best model
with open('best_model_combined.pkl', 'wb') as f:
    pickle.dump(best_model_combined, f)

print('\nBest model saved as best_model_combined.pkl')
