import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("MAESTRO - Mass Prediction Model")
print("="*80)

# ============================================================================
# STEP 1: Load the data
# ============================================================================
print("\n[1] Loading data from Buzzard_DC1.csv...")
df = pd.read_csv('Buzzard_DC1.csv')

print(f"Data loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Optimize data types - use float32 for features to save memory
print(f"\n[1.1] Optimizing data types...")
print(f"  Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Convert all float64 columns to float32
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')

print(f"  Optimized memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  Memory saved: {(df.memory_usage(deep=True).sum() / 1024**2) * 0.5:.2f} MB (~50% reduction)")
print(f"  All features now using float32 (sufficient precision for ML)")

# ============================================================================
# STEP 2: Display first 5 rows
# ============================================================================
print("\n[2] First 5 rows of the dataset:")
print("-" * 80)
print(df.head())

print("\n[3] Column names and data types:")
print("-" * 80)
print(df.dtypes)

print("\n[4] Basic statistics:")
print("-" * 80)
print(df.describe())

# ============================================================================
# STEP 3: Analyze distributions and identify outliers
# ============================================================================
print("\n" + "="*80)
print("OUTLIER DETECTION ANALYSIS")
print("="*80)

# Define the columns we'll be working with
feature_columns = ['u', 'g', 'r', 'i', 'z', 'y', 
                   'u.err', 'g.err', 'r.err', 'i.err', 'z.err', 'y.err', 
                   'redshift']
target_column = 'log.mass'
all_columns = feature_columns + [target_column]

print(f"\n[5] Analyzing distributions for {len(all_columns)} variables...")

# Check for missing values first
print("\n[6] Missing values check:")
print("-" * 80)
missing_counts = df[all_columns].isnull().sum()
if missing_counts.sum() == 0:
    print("[OK] No missing values found in the dataset")
else:
    print(missing_counts[missing_counts > 0])

# ============================================================================
# Distribution Analysis
# ============================================================================
print("\n[7] Distribution Analysis:")
print("-" * 80)

# Create distribution plots for all variables
n_cols = 4
n_rows = int(np.ceil(len(all_columns) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(all_columns):
    ax = axes[idx]
    
    # Plot histogram with KDE
    df[col].hist(bins=50, alpha=0.7, ax=ax, edgecolor='black')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {col}')
    
    # Add vertical lines for mean and median
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.legend(fontsize=8)
    
    # Calculate skewness
    skewness = stats.skew(df[col])
    ax.text(0.02, 0.98, f'Skew: {skewness:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove empty subplots
for idx in range(len(all_columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
print("[OK] Distribution plots saved as 'distributions.png'")

# ============================================================================
# Outlier Detection using Multiple Methods
# ============================================================================
print("\n[8] Outlier Detection Methods:")
print("-" * 80)

outlier_results = {}

# Method 1: Z-Score (for normally distributed data)
print("\nMethod 1: Z-Score Analysis (threshold = 3)")
print("-" * 40)
z_scores = np.abs(stats.zscore(df[all_columns]))
z_score_outliers = (z_scores > 3).any(axis=1)
outlier_results['z_score'] = z_score_outliers
print(f"  Outliers detected: {z_score_outliers.sum()} ({z_score_outliers.sum()/len(df)*100:.2f}%)")

# Method 2: IQR (Interquartile Range) - robust to skewed data
print("\nMethod 2: IQR (Interquartile Range)")
print("-" * 40)
iqr_outliers = pd.Series([False] * len(df), index=df.index)

for col in all_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    iqr_outliers |= col_outliers
    print(f"  {col:12s}: {col_outliers.sum():6d} outliers (bounds: [{lower_bound:.4f}, {upper_bound:.4f}])")

outlier_results['iqr'] = iqr_outliers
print(f"\nTotal unique outliers by IQR: {iqr_outliers.sum()} ({iqr_outliers.sum()/len(df)*100:.2f}%)")

# Method 3: Modified Z-Score (using MAD - Median Absolute Deviation)
# More robust than regular z-score for skewed distributions
print("\nMethod 3: Modified Z-Score (MAD-based, threshold = 3.5)")
print("-" * 40)

def modified_z_score(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros(len(data))
    return np.abs(modified_z_scores)

mad_outliers = pd.Series([False] * len(df), index=df.index)
for col in all_columns:
    mod_z_scores = modified_z_score(df[col].values)
    col_outliers = mod_z_scores > 3.5
    mad_outliers |= col_outliers
    print(f"  {col:12s}: {col_outliers.sum():6d} outliers")

outlier_results['mad'] = mad_outliers
print(f"\nTotal unique outliers by MAD: {mad_outliers.sum()} ({mad_outliers.sum()/len(df)*100:.2f}%)")

# ============================================================================
# Consensus Outliers
# ============================================================================
print("\n[9] Consensus Analysis:")
print("-" * 80)

# Count how many methods flag each row as an outlier
outlier_counts = sum(outlier_results.values())

print("\nOutlier agreement across methods:")
for i in range(len(outlier_results) + 1):
    count = (outlier_counts == i).sum()
    percentage = count / len(df) * 100
    print(f"  Flagged by {i}/3 methods: {count:7d} rows ({percentage:5.2f}%)")

# Define consensus outliers (flagged by at least 2 methods)
consensus_outliers = outlier_counts >= 2
print(f"\n[OK] Consensus outliers (flagged by >=2 methods): {consensus_outliers.sum()} rows ({consensus_outliers.sum()/len(df)*100:.2f}%)")

# ============================================================================
# Create Box Plots for Visual Outlier Detection
# ============================================================================
print("\n[10] Creating box plots for visual outlier detection...")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(all_columns):
    ax = axes[idx]
    bp = ax.boxplot(df[col], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('Value')
    ax.set_title(f'Box Plot: {col}')
    ax.grid(True, alpha=0.3)

# Remove empty subplots
for idx in range(len(all_columns), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
print("[OK] Box plots saved as 'boxplots.png'")

# ============================================================================
# Save outlier information
# ============================================================================
print("\n[11] Saving outlier information...")

# Add outlier flags to dataframe
df_with_flags = df.copy()
df_with_flags['outlier_z_score'] = outlier_results['z_score']
df_with_flags['outlier_iqr'] = outlier_results['iqr']
df_with_flags['outlier_mad'] = outlier_results['mad']
df_with_flags['outlier_count'] = outlier_counts
df_with_flags['is_consensus_outlier'] = consensus_outliers

# Save to CSV
df_with_flags.to_csv('data_with_outlier_flags.csv', index=False)
print("[OK] Data with outlier flags saved as 'data_with_outlier_flags.csv'")

# Save only the outliers
df_outliers = df_with_flags[consensus_outliers]
df_outliers.to_csv('consensus_outliers.csv', index=False)
print(f"[OK] Consensus outliers saved as 'consensus_outliers.csv' ({len(df_outliers)} rows)")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

print(f"\nDataset Overview:")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Features for modeling: {len(feature_columns)}")
print(f"  Target variable: {target_column}")

print(f"\nOutlier Detection Results:")
print(f"  Z-Score method:     {outlier_results['z_score'].sum():7,} outliers ({outlier_results['z_score'].sum()/len(df)*100:5.2f}%)")
print(f"  IQR method:         {outlier_results['iqr'].sum():7,} outliers ({outlier_results['iqr'].sum()/len(df)*100:5.2f}%)")
print(f"  MAD method:         {outlier_results['mad'].sum():7,} outliers ({outlier_results['mad'].sum()/len(df)*100:5.2f}%)")
print(f"  Consensus (≥2):     {consensus_outliers.sum():7,} outliers ({consensus_outliers.sum()/len(df)*100:5.2f}%)")

print(f"\nClean data (non-outliers): {(~consensus_outliers).sum():,} rows ({(~consensus_outliers).sum()/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("Analysis complete! Check the output files:")
print("  - distributions.png: Distribution plots for all variables")
print("  - boxplots.png: Box plots for outlier visualization")
print("  - data_with_outlier_flags.csv: Original data with outlier flags")
print("  - consensus_outliers.csv: Rows identified as outliers by multiple methods")
print("="*80)

# ============================================================================
# RANDOM FOREST MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("RANDOM FOREST MODEL TRAINING")
print("="*80)

# Prepare datasets - with and without outliers
df_clean = df_with_flags[~consensus_outliers].copy()
df_all = df_with_flags.copy()

print(f"\n[12] Dataset Preparation:")
print("-" * 80)
print(f"  Full dataset:  {len(df_all):,} rows")
print(f"  Clean dataset: {len(df_clean):,} rows (outliers removed)")
print(f"  Features:      {feature_columns}")
print(f"  Target:        {target_column}")

# Function to perform train/test split with stratification
def stratified_split_regression(df, features, target, test_size=0.3, random_state=42):
    """
    Perform stratified train/test split for regression by binning the target variable.
    This ensures similar distribution of the target in both train and test sets.
    Default split: 70% train, 30% test
    """
    # Create bins for stratification (10 bins based on percentiles)
    n_bins = 10
    df_temp = df.copy()
    df_temp['target_bin'] = pd.qcut(df_temp[target], q=n_bins, labels=False, duplicates='drop')
    
    # Perform stratified split
    X = df_temp[features]
    y = df_temp[target]
    stratify_labels = df_temp['target_bin']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_labels, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate model
def train_and_evaluate_model(df, dataset_name, features, target, test_size=0.3, random_state=42):
    """
    Train a Random Forest model and evaluate its performance.
    Default: 70/30 train/test split
    """
    print(f"\n{'='*80}")
    print(f"Training on: {dataset_name}")
    print('='*80)
    
    # Split the data
    print(f"\n[Split] Performing stratified train/test split (test_size={test_size})...")
    X_train, X_test, y_train, y_test = stratified_split_regression(
        df, features, target, test_size=test_size, random_state=random_state
    )
    
    print(f"  Train set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Test set:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Check distribution similarity
    print(f"\n[Distribution] Target variable statistics:")
    print(f"  Full dataset  - Mean: {df[target].mean():.4f}, Std: {df[target].std():.4f}")
    print(f"  Train set     - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    print(f"  Test set      - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    
    # Train Random Forest model
    print(f"\n[Model] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    print(f"  Model trained successfully!")
    
    # Make predictions
    print(f"\n[Predictions] Generating predictions...")
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n[Results] Model Performance:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f} {abs(train_r2-test_r2):<15.4f}")
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f} {abs(train_rmse-test_rmse):<15.4f}")
    print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f} {abs(train_mae-test_mae):<15.4f}")
    print("-" * 80)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n[Feature Importance] Top features:")
    print("-" * 80)
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:<15s}: {row['importance']:.6f} {'█' * int(row['importance'] * 100)}")
    
    # Create prediction plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set
    axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=10)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual log.mass')
    axes[0].set_ylabel('Predicted log.mass')
    axes[0].set_title(f'Training Set (R²={train_r2:.4f}, RMSE={train_rmse:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual log.mass')
    axes[1].set_ylabel('Predicted log.mass')
    axes[1].set_title(f'Test Set (R²={test_r2:.4f}, RMSE={test_rmse:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f'predictions_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Prediction plots saved as '{plot_filename}'")
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance.plot(x='feature', y='importance', kind='barh', ax=ax, legend=False)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance - {dataset_name}')
    plt.tight_layout()
    importance_filename = f'feature_importance_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Feature importance plot saved as '{importance_filename}'")
    
    # Residual plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    axes[0].scatter(y_train_pred, train_residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted log.mass')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Training Set Residuals')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test_pred, test_residuals, alpha=0.3, s=10, color='green')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted log.mass')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'Test Set Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    residual_filename = f'residuals_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(residual_filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Residual plots saved as '{residual_filename}'")
    
    return {
        'model': rf_model,
        'feature_importance': feature_importance,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# Train models on both datasets
print("\n" + "="*80)
print("Training Model #1: Clean Dataset (Outliers Removed)")
print("="*80)
results_clean = train_and_evaluate_model(
    df_clean, 
    "Clean Dataset", 
    feature_columns, 
    target_column,
    test_size=0.3,
    random_state=42
)

print("\n" + "="*80)
print("Training Model #2: Full Dataset (Including Outliers)")
print("="*80)
results_full = train_and_evaluate_model(
    df_all, 
    "Full Dataset", 
    feature_columns, 
    target_column,
    test_size=0.3,
    random_state=42
)

# ============================================================================
# Final Comparison
# ============================================================================
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Dataset': ['Clean (No Outliers)', 'Full (With Outliers)'],
    'Train_Size': [len(results_clean['X_train']), len(results_full['X_train'])],
    'Test_Size': [len(results_clean['X_test']), len(results_full['X_test'])],
    'Train_R2': [results_clean['train_r2'], results_full['train_r2']],
    'Test_R2': [results_clean['test_r2'], results_full['test_r2']],
    'Train_RMSE': [results_clean['train_rmse'], results_full['train_rmse']],
    'Test_RMSE': [results_clean['test_rmse'], results_full['test_rmse']],
    'Overfit_Gap': [
        abs(results_clean['train_r2'] - results_clean['test_r2']),
        abs(results_full['train_r2'] - results_full['test_r2'])
    ]
})

print("\n")
print(comparison_df.to_string(index=False))

# Determine best model
if results_clean['test_r2'] > results_full['test_r2']:
    print(f"\n[RECOMMENDATION] Use the CLEAN dataset model (better test R² score)")
    best_model = results_clean['model']
    best_name = "clean_dataset"
else:
    print(f"\n[RECOMMENDATION] Use the FULL dataset model (better test R² score)")
    best_model = results_full['model']
    best_name = "full_dataset"

# Save the best model
import joblib
model_filename = f'rf_model_{best_name}.joblib'
joblib.dump(best_model, model_filename)
print(f"[OK] Best model saved as '{model_filename}'")

print("\n" + "="*80)
print("ALL TASKS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - distributions.png: Distribution plots for all variables")
print("  - boxplots.png: Box plots for outlier visualization")
print("  - data_with_outlier_flags.csv: Original data with outlier flags")
print("  - consensus_outliers.csv: Rows identified as outliers")
print("  - predictions_clean_dataset.png: Predictions for clean model")
print("  - predictions_full_dataset.png: Predictions for full model")
print("  - feature_importance_clean_dataset.png: Feature importance (clean)")
print("  - feature_importance_full_dataset.png: Feature importance (full)")
print("  - residuals_clean_dataset.png: Residual analysis (clean)")
print("  - residuals_full_dataset.png: Residual analysis (full)")
print(f"  - {model_filename}: Best trained model")
print("="*80)

