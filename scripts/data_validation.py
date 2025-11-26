import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataValidation:

    def __init__(self, DATA_PATH, TARGET_COL):
        self.DATA_PATH = pathlib.Path(DATA_PATH)
        self.TARGET_COL = TARGET_COL

    # -------------------------------------------
    # Loading
    # -------------------------------------------
    def _load_data(self):
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found at {self.DATA_PATH}")
        return pd.read_csv(self.DATA_PATH)

    # -------------------------------------------
    # Validation checks
    # -------------------------------------------
    def _check_file_format(self):
        if self.DATA_PATH.suffix.lower() != ".csv":
            raise ValueError(f"Expected .csv file, got {self.DATA_PATH.suffix}")

        if self.DATA_PATH.stat().st_size == 0:
            raise ValueError("The CSV file is empty.")

        try:
            pd.read_csv(self.DATA_PATH, nrows=5)
        except Exception as e:
            raise ValueError(f"File is not a valid CSV or cannot be parsed: {e}")

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

    def _check_target_distribution_full(self, df):
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

    def _check_target_feature_correlations(self, train_df, threshold=0.99):
        y_encoded = train_df[self.TARGET_COL].map({'Y': 1, 'N': 0})
        if y_encoded.isna().any():
            raise ValueError("Target must contain only 'Y'/'N' for encoding.")

        numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        temp = train_df[numeric_features].copy()
        temp["_target_"] = y_encoded

        corrs = temp.corr()["_target_"].drop("_target_")
        suspicious = corrs[abs(corrs) > threshold]

        if len(suspicious) > 0:
            raise ValueError(
                "High target–feature correlations detected: "
                f"{suspicious.to_dict()}"
            )

    def _check_feature_correlations(self, train_df, threshold=0.99):
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return

        corr_matrix = train_df[numeric_cols].corr().abs()
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

    # -------------------------------------------
    # Public API
    # -------------------------------------------
    def validate_dataset(self):
        self._check_file_format()
        df = self._load_data()

        # Modify these per project
        expected_columns = [
            "Customer_ID", "Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Applicant_Income", "Coapplicant_Income",
            "Loan_Amount", "Loan_Amount_Term", "Credit_History",
            "Property_Area", "Loan_Status"
        ]

        expected_types = {
            "Customer_ID": "int",
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
            "Loan_Status": "object"
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
            "Loan_Status": {"Y", "N"}
        }

        # --------- Run the checks ---------
        self._check_column_names(df, expected_columns)
        self._check_no_empty_rows(df)
        self._check_missingness_threshold(df)
        self._check_data_types(df, expected_types)
        self._check_no_duplicates(df)
        self._check_outliers(df, range_rules=range_rules, zscore_threshold=20.0)
        self._check_category_levels(df, expected_category_levels)
        self._check_target_distribution_full(df)

        # Split dataset
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)

        self._check_target_feature_correlations(train_df)
        self._check_feature_correlations(train_df)

        print("-----Data validation completed successfully.-----")