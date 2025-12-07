"""
Process and validate loan eligibility data.

This script:
1. Loads raw data
2. Handles missing values
3. Splits into train/test sets
4. Encodes categorical variables
5. Scales numerical features
6. Validates data quality
7. Saves processed datasets

Usage:
    python scripts/process_data.py --input-path PATH --output-dir DIR
"""

import os
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_col='Loan_Status', test_size=0.2, random_state=522):
    """
    Preprocess loan eligibility data with one-hot encoding and return train/test sets
    along with df_train and df_test BEFORE scaling and encoding.
    """
    # Make a copy
    data = df.copy()

    # Drop Customer_ID
    if 'Customer_ID' in data.columns:
        data = data.drop('Customer_ID', axis=1)

    # Handle missing values
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = [c for c in numerical_cols if c != target_col]
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())

    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_cols = [c for c in categorical_cols if c != target_col]
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Separate X and y
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Encode target (Y/N → 1/0)
    y = (y == 'Y').astype(int)

    # -------- SPLIT BEFORE ENCODING/SCALING --------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save df_train & df_test BEFORE transformations
    df_train = pd.concat([X_train_raw.reset_index(drop=True),
                        y_train.reset_index(drop=True)], axis=1)

    df_test = pd.concat([X_test_raw.reset_index(drop=True),
                        y_test.reset_index(drop=True)], axis=1)

    # One-hot encode using training columns only
    X_train_encoded = pd.get_dummies(X_train_raw, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=categorical_cols, drop_first=True)

    # Align columns (important!)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    feature_names = X_train_encoded.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    return (
        X_train_scaled, X_test_scaled, 
        y_train, y_test,
        feature_names, scaler,
        df_train, df_test          
    )


class DataValidation:
    """Validates processed training data to ensure quality and consistency."""

    def __init__(self, df_train: pd.DataFrame, target_col: str):
        """Initialize DataValidation with training dataframe."""
        self.df = df_train.copy()
        self.TARGET_COL = target_col

    def _check_column_names(self, df, expected_cols):
        actual = set(df.columns)
        expected = set(expected_cols)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if extra:
            raise ValueError(f"Unexpected extra columns: {extra}")

    def _check_no_empty_rows(self, df):
        n_empty = df.isna().all(axis=1).sum()
        if n_empty > 0:
            raise ValueError(f"Found {n_empty} completely empty rows.")

    def _check_missingness_threshold(self, df, max_missing_fraction=0.3):
        missing_fraction = df.isna().mean()
        too_missing = missing_fraction[missing_fraction > max_missing_fraction]
        if not too_missing.empty:
            raise ValueError(
                f"Columns exceeding missing threshold ({max_missing_fraction*100}%): "
                f"{too_missing.to_dict()}"
            )

    def _check_data_types(self, df, expected_types):
        for col, expected_type in expected_types.items():
            actual_type = df[col].dtype
            if expected_type == "int" and not np.issubdtype(actual_type, np.integer):
                raise TypeError(f"Column '{col}' must be integer, got {actual_type}")
            if expected_type == "float" and not np.issubdtype(actual_type, np.floating):
                raise TypeError(f"Column '{col}' must be float, got {actual_type}")
            if expected_type == "object" and actual_type != "object":
                raise TypeError(f"Column '{col}' must be object (string), got {actual_type}")

    def _check_no_duplicates(self, df):
        n_dup = df.duplicated().sum()
        if n_dup > 0:
            raise ValueError(f"Dataset contains {n_dup} duplicate rows.")

    def _check_outliers(self, df, range_rules=None, zscore_threshold=5.0):
        if range_rules:
            for col, (min_val, max_val) in range_rules.items():
                if min_val is not None and (df[col] < min_val).any():
                    raise ValueError(f"{col} contains values below {min_val}")
                if max_val is not None and (df[col] > max_val).any():
                    raise ValueError(f"{col} contains values above {max_val}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            zscores = (df[col] - mean) / std
            if (abs(zscores) > zscore_threshold).any():
                raise ValueError(f"{col} contains statistical outliers (z > {zscore_threshold}).")

    def _check_category_levels(self, df, expected_levels):
        for col, allowed in expected_levels.items():
            actual = set(df[col].dropna().unique())
            unexpected = actual - set(allowed)
            if unexpected:
                raise ValueError(f"{col} contains unexpected categories: {unexpected}")
            counts = df[col].value_counts()
            rare = counts[counts == 1]
            if len(rare) > 0:
                raise ValueError(
                    f"{col} has extremely rare categories (only 1 occurrence): {list(rare.index)}"
                )

    def _check_target_distribution(self, df):
        y = df[self.TARGET_COL]
        if y.isna().any():
            raise ValueError("Target column contains missing values.")
        if y.nunique() < 2:
            raise ValueError("Target column must have at least 2 unique values.")
        counts = y.value_counts(normalize=True)
        if counts.max() > 0.99:
            raise ValueError(
                f"Target distribution extremely imbalanced: {counts.max()*100:.1f}% in one class."
            )

    def _check_target_feature_correlations(self, df, threshold=0.99):
        y = df[self.TARGET_COL]
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Target column '{self.TARGET_COL}' must be numeric.")
        if y.isna().any():
            raise ValueError("Target column contains missing values.")
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col != self.TARGET_COL]
        temp = df[numeric_features].copy()
        temp["_target_"] = y
        corrs = temp.corr()["_target_"].drop("_target_")
        suspicious = corrs[abs(corrs) > threshold]
        if len(suspicious) > 0:
            raise ValueError(
                f"High target–feature correlations detected: {suspicious.to_dict()}"
            )
        
    def _check_feature_correlations(self, df, threshold=0.99):
        df_X = df.copy().drop(self.TARGET_COL, axis=1)
        numeric_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return
        corr_matrix = df_X[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        suspicious_pairs = [
            (a, b, upper.loc[a, b])
            for a in upper.columns
            for b in upper.columns
            if upper.loc[a, b] > threshold
        ]
        if suspicious_pairs:
            formatted = {f"{a}-{b}": float(r) for a, b, r in suspicious_pairs}
            raise ValueError(
                f"High feature–feature correlations detected: {formatted}"
            )

    def validate_dataset(self):
        """Run all validation checks."""
        df = self.df
        expected_columns = [
            "Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Applicant_Income", "Coapplicant_Income",
            "Loan_Amount", "Loan_Amount_Term", "Credit_History",
            "Property_Area", "Loan_Status"
        ]
        expected_types = {
            "Gender": "object",
            "Married": "object",
            "Dependents": "int",
            "Education": "object",
            "Self_Employed": "object",
            "Applicant_Income": "int",
            "Coapplicant_Income": "float",
            "Loan_Amount": "int",
            "Loan_Amount_Term": "int",
            "Credit_History": "int",
            "Property_Area": "object",
            "Loan_Status": "int"
        }
        range_rules = {
            "Dependents": (0, 20),
            "Loan_Amount_Term": (0, None)
        }
        expected_category_levels = {
            "Gender": {"Male", "Female"},
            "Married": {"No", "Yes"},
            "Education": {"Graduate", "Not Graduate"},
            "Self_Employed": {"No", "Yes"},
            "Property_Area": {"Urban", "Semiurban", "Rural"},
            "Loan_Status": {0, 1}
        }
        self._check_column_names(df, expected_columns)
        self._check_no_empty_rows(df)
        self._check_missingness_threshold(df)
        self._check_data_types(df, expected_types)
        self._check_no_duplicates(df)
        self._check_outliers(df, range_rules=range_rules, zscore_threshold=20.0)
        self._check_category_levels(df, expected_category_levels)
        self._check_target_distribution(df)
        self._check_target_feature_correlations(df)
        self._check_feature_correlations(df)
        print(" Training dataset validation completed successfully")


@click.command()
@click.option(
    "--input-path",
    type=str,
    required=True,
    help="Path to the raw dataset CSV file"
)
@click.option(
    "--output-dir",
    type=str,
    default="data/processed",
    help="Directory where processed and split data will be saved"
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for test set (default: 0.2)"
)
@click.option(
    "--random-state",
    type=int,
    default=522,
    help="Random seed for reproducibility (default: 522)"
)
def main(input_path, output_dir, test_size, random_state):
    """
    Process raw loan data: clean, validate, split, and scale.
    
    This script:
    1. Loads raw data from CSV
    2. Handles missing values (median for numeric, mode for categorical)
    3. Splits data into train/test sets with stratification
    4. One-hot encodes categorical variables
    5. Scales numerical features using StandardScaler
    6. Validates data quality
    7. Saves all processed datasets
    
    Examples:
        # With default settings
        python scripts/process_data.py \\
            --input-path "data/raw/Loan Eligibility Prediction.csv"
        
        # With custom settings
        python scripts/process_data.py \\
            --input-path "data/raw/Loan Eligibility Prediction.csv" \\
            --output-dir data/processed \\
            --test-size 0.2 \\
            --random-state 522
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading raw data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")

    target_col = "Loan_Status"

    # Preprocess + split data
    print("\nRunning preprocessing pipeline...")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, df_train, df_test = (
        preprocess_data(
            df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state
        )
    )
    
    print(f"  Train size: {len(df_train)} samples")
    print(f"  Test size: {len(df_test)} samples")
    print(f"  Features after encoding: {len(feature_names)}")
    
    # Data validation
    print("\nRunning data validation...")
    validator = DataValidation(df_train, target_col)
    validator.validate_dataset()

    # Save outputs to CSV
    print("\nSaving processed datasets...")
    pd.DataFrame(X_train_scaled, columns=feature_names) \
        .to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    pd.DataFrame(X_test_scaled, columns=feature_names) \
        .to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)
    pd.DataFrame(y_train, columns=[target_col]) \
        .to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=[target_col]) \
        .to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    df_train.to_csv(os.path.join(output_dir, "df_train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "df_test.csv"), index=False)

    print(f"\nAll processed data saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - X_train_scaled.csv (scaled training features)")
    print("  - X_test_scaled.csv (scaled test features)")
    print("  - y_train.csv (training labels)")
    print("  - y_test.csv (test labels)")
    print("  - df_train.csv (unscaled training data for EDA)")
    print("  - df_test.csv (unscaled test data)")


if __name__ == "__main__":
    main()