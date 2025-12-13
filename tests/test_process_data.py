"""
Tests for process_data.py

Run with: pytest tests/test_process_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from click.testing import CliRunner
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.process_data import main, preprocess_data, DataValidation


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_raw_data():
    """Create sample raw loan data DataFrame."""
    np.random.seed(42)
    n_samples = 200
    
    return pd.DataFrame({
        'Customer_ID': [f'CUST{i:04d}' for i in range(n_samples)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.randint(0, 4, n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'Applicant_Income': np.random.randint(1500, 80000, n_samples),
        'Coapplicant_Income': np.random.uniform(0, 50000, n_samples),
        'Loan_Amount': np.random.randint(9, 700, n_samples),
        'Loan_Amount_Term': np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
        'Loan_Status': np.random.choice(['Y', 'N'], n_samples)
    })


@pytest.fixture
def sample_raw_data_with_missing():
    """Create sample raw data with missing values."""
    data = pd.DataFrame({
        'Customer_ID': ['CUST0001', 'CUST0002', 'CUST0003', 'CUST0004'],
        'Gender': ['Male', 'Female', None, 'Male'],
        'Married': ['Yes', None, 'No', 'Yes'],
        'Dependents': [0, 1, None, 2],
        'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes', None, 'No'],
        'Applicant_Income': [5000, None, 6000, 7000],
        'Coapplicant_Income': [2000, 3000, None, 0],
        'Loan_Amount': [100, None, 150, 200],
        'Loan_Amount_Term': [360, 360, None, 180],
        'Credit_History': [1, None, 1, 0],
        'Property_Area': ['Urban', 'Rural', 'Semiurban', None],
        'Loan_Status': ['Y', 'N', 'Y', 'N']
    })
    return data


@pytest.fixture
def sample_validated_train_data():
    """Create sample DataFrame that passes validation."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.randint(0, 4, n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'Applicant_Income': np.random.randint(1500, 80000, n_samples),
        'Coapplicant_Income': np.random.uniform(0, 50000, n_samples),
        'Loan_Amount': np.random.randint(9, 700, n_samples),
        'Loan_Amount_Term': np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
        'Loan_Status': np.random.choice([0, 1], n_samples)
    })


class TestPreprocessDataFunction:
    """Test suite for preprocess_data function."""

    def test_preprocess_data_basic(self, sample_raw_data):
        """Test basic preprocessing functionality."""
        result = preprocess_data(sample_raw_data, test_size=0.2, random_state=42)
        
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, df_train, df_test = result
        
        # Check return values
        assert isinstance(X_train_scaled, pd.DataFrame)
        assert isinstance(X_test_scaled, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(feature_names, list)
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

    def test_preprocess_data_drops_customer_id(self, sample_raw_data):
        """Test that Customer_ID is dropped from processed data."""
        result = preprocess_data(sample_raw_data)
        X_train_scaled, _, _, _, _, _, df_train, _ = result
        
        assert 'Customer_ID' not in df_train.columns
        assert 'Customer_ID' not in X_train_scaled.columns

    def test_preprocess_data_handles_missing_values(self, sample_raw_data_with_missing):
        """Test that missing values are handled."""
        result = preprocess_data(sample_raw_data_with_missing, test_size=0.5, random_state=42)
        X_train_scaled, _, _, _, _, _, df_train, _ = result
        
        # Check that no missing values in processed data
        assert df_train.isna().sum().sum() == 0
        assert X_train_scaled.isna().sum().sum() == 0

    def test_preprocess_data_encodes_target(self, sample_raw_data):
        """Test that target is encoded from Y/N to 1/0."""
        result = preprocess_data(sample_raw_data, random_state=42)
        _, _, y_train, y_test, _, _, df_train, _ = result
        
        # Check target encoding
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
        assert set(df_train['Loan_Status'].unique()).issubset({0, 1})

    def test_preprocess_data_train_test_split(self, sample_raw_data):
        """Test that train/test split works correctly."""
        result = preprocess_data(sample_raw_data, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, y_train, y_test, _, _, df_train, df_test = result
        
        # Check split sizes
        total_samples = len(df_train) + len(df_test)
        assert abs(len(df_test) / total_samples - 0.2) < 0.05  # Allow small rounding error
        assert len(X_train_scaled) == len(df_train)
        assert len(X_test_scaled) == len(df_test)
        assert len(y_train) == len(df_train)
        assert len(y_test) == len(df_test)

    def test_preprocess_data_one_hot_encoding(self, sample_raw_data):
        """Test that categorical variables are one-hot encoded."""
        result = preprocess_data(sample_raw_data, random_state=42)
        X_train_scaled, X_test_scaled, _, _, feature_names, _, _, _ = result
        
        # Check that categorical columns are encoded
        original_categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        for col in original_categorical:
            # Original column should not be in feature names (or should have prefix)
            assert col not in feature_names or any(f.startswith(col + '_') for f in feature_names)

    def test_preprocess_data_scaling(self, sample_raw_data):
        """Test that features are scaled."""
        result = preprocess_data(sample_raw_data, random_state=42)
        X_train_scaled, X_test_scaled, _, _, _, _, _, _ = result
        
        # Check that scaled data has approximately mean 0 and std 1 for training
        train_means = X_train_scaled.mean()
        train_stds = X_train_scaled.std()
        
        # Allow small tolerance
        assert (train_means.abs() < 1e-10).all()  # Mean should be ~0
        assert (train_stds.abs() - 1.0 < 1e-10).all()  # Std should be ~1

    def test_preprocess_data_feature_alignment(self, sample_raw_data):
        """Test that train and test have same features."""
        result = preprocess_data(sample_raw_data, random_state=42)
        X_train_scaled, X_test_scaled, _, _, feature_names, _, _, _ = result
        
        # Check feature alignment
        assert list(X_train_scaled.columns) == feature_names
        assert list(X_test_scaled.columns) == feature_names

    def test_preprocess_data_reproducibility(self, sample_raw_data):
        """Test that same random_state produces same results."""
        result1 = preprocess_data(sample_raw_data, random_state=42)
        result2 = preprocess_data(sample_raw_data, random_state=42)
        
        X_train1, _, y_train1, _, _, _, _, _ = result1
        X_train2, _, y_train2, _, _, _, _, _ = result2
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2, check_names=False)

    def test_preprocess_data_custom_test_size(self, sample_raw_data):
        """Test custom test_size parameter."""
        result = preprocess_data(sample_raw_data, test_size=0.3, random_state=42)
        _, _, _, _, _, _, df_train, df_test = result
        
        total = len(df_train) + len(df_test)
        assert abs(len(df_test) / total - 0.3) < 0.05


class TestDataValidationClass:
    """Test suite for DataValidation class."""

    def test_validation_passes_valid_data(self, sample_validated_train_data):
        """Test that valid data passes validation."""
        validator = DataValidation(sample_validated_train_data, 'Loan_Status')
        # Should not raise any exceptions
        validator.validate_dataset()

    def test_validation_missing_columns_error(self, sample_validated_train_data):
        """Test that missing columns raise error."""
        invalid_data = sample_validated_train_data.drop('Gender', axis=1)
        validator = DataValidation(invalid_data, 'Loan_Status')
        
        with pytest.raises(ValueError, match="Missing columns"):
            validator.validate_dataset()

    def test_validation_empty_rows_error(self):
        """Test that completely empty rows raise error."""
        data = pd.DataFrame({
            'Gender': ['Male', np.nan],
            'Married': ['Yes', np.nan],
            'Loan_Status': [1, np.nan]
        })
        # Add required columns with NaN
        for col in ['Dependents', 'Education', 'Self_Employed', 'Applicant_Income',
                    'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
                    'Credit_History', 'Property_Area']:
            data[col] = np.nan
        
        validator = DataValidation(data, 'Loan_Status')
        
        with pytest.raises(ValueError, match="completely empty rows"):
            validator.validate_dataset()

    def test_validation_wrong_data_types_error(self, sample_validated_train_data):
        """Test that wrong data types raise error."""
        invalid_data = sample_validated_train_data.copy()
        invalid_data['Dependents'] = invalid_data['Dependents'].astype(float)
        validator = DataValidation(invalid_data, 'Loan_Status')
        
        with pytest.raises(TypeError, match="must be integer"):
            validator.validate_dataset()

    def test_validation_duplicate_rows_error(self, sample_validated_train_data):
        """Test that duplicate rows raise error."""
        invalid_data = pd.concat([sample_validated_train_data, sample_validated_train_data.iloc[[0]]])
        validator = DataValidation(invalid_data, 'Loan_Status')
        
        with pytest.raises(ValueError, match="duplicate rows"):
            validator.validate_dataset()

    def test_validation_unexpected_categories_error(self, sample_validated_train_data):
        """Test that unexpected categories raise error."""
        invalid_data = sample_validated_train_data.copy()
        invalid_data.loc[0, 'Gender'] = 'Other'
        validator = DataValidation(invalid_data, 'Loan_Status')
        
        with pytest.raises(ValueError, match="unexpected categories"):
            validator.validate_dataset()

    def test_validation_target_imbalance_error(self):
        """Test that extremely imbalanced target raises error."""
        data = pd.DataFrame({
            'Gender': ['Male'] * 100,
            'Married': ['Yes'] * 100,
            'Dependents': [0] * 100,
            'Education': ['Graduate'] * 100,
            'Self_Employed': ['No'] * 100,
            'Applicant_Income': [5000] * 100,
            'Coapplicant_Income': [0.0] * 100,
            'Loan_Amount': [100] * 100,
            'Loan_Amount_Term': [360] * 100,
            'Credit_History': [1] * 100,
            'Property_Area': ['Urban'] * 100,
            'Loan_Status': [1] * 99 + [0]
        })
        validator = DataValidation(data, 'Loan_Status')
        
        with pytest.raises(ValueError, match="extremely imbalanced"):
            validator.validate_dataset()

    def test_validation_target_missing_error(self, sample_validated_train_data):
        """Test that missing target values raise error."""
        invalid_data = sample_validated_train_data.copy()
        invalid_data.loc[0, 'Loan_Status'] = np.nan
        validator = DataValidation(invalid_data, 'Loan_Status')
        
        with pytest.raises(ValueError, match="contains missing values"):
            validator.validate_dataset()


class TestProcessDataCLI:
    """Test suite for process_data CLI."""

    def test_cli_basic_execution(self, runner, tmp_path, sample_raw_data):
        """Test that CLI runs successfully with valid inputs."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        result = runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "processed data saved" in result.output.lower()

    def test_cli_creates_output_directory(self, runner, tmp_path, sample_raw_data):
        """Test that output directory is created."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "new_output"
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        assert output_dir.exists()

    def test_cli_saves_all_output_files(self, runner, tmp_path, sample_raw_data):
        """Test that all expected output files are created."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        assert (output_dir / "X_train_scaled.csv").exists()
        assert (output_dir / "X_test_scaled.csv").exists()
        assert (output_dir / "y_train.csv").exists()
        assert (output_dir / "y_test.csv").exists()
        assert (output_dir / "df_train.csv").exists()
        assert (output_dir / "df_test.csv").exists()

    def test_cli_missing_input_file_error(self, runner, tmp_path):
        """Test error when input file is missing."""
        result = runner.invoke(main, [
            '--input-path', 'nonexistent.csv',
            '--output-dir', str(tmp_path)
        ])
        
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_cli_custom_test_size(self, runner, tmp_path, sample_raw_data):
        """Test that custom test_size parameter works."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        result = runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir),
            '--test-size', '0.3'
        ])
        
        assert result.exit_code == 0
        df_test = pd.read_csv(output_dir / "df_test.csv")
        df_train = pd.read_csv(output_dir / "df_train.csv")
        total = len(df_test) + len(df_train)
        assert abs(len(df_test) / total - 0.3) < 0.05

    def test_cli_custom_random_state(self, runner, tmp_path, sample_raw_data):
        """Test that custom random_state parameter works."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir1 = tmp_path / "output1"
        output_dir2 = tmp_path / "output2"
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir1),
            '--random-state', '42'
        ])
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir2),
            '--random-state', '42'
        ])
        
        y_train1 = pd.read_csv(output_dir1 / "y_train.csv")
        y_train2 = pd.read_csv(output_dir2 / "y_train.csv")
        pd.testing.assert_frame_equal(y_train1, y_train2)


class TestProcessDataOutputs:
    """Test suite for output file formats and contents."""

    def test_output_files_readable(self, runner, tmp_path, sample_raw_data):
        """Test that all output files can be read."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        X_train = pd.read_csv(output_dir / "X_train_scaled.csv")
        X_test = pd.read_csv(output_dir / "X_test_scaled.csv")
        y_train = pd.read_csv(output_dir / "y_train.csv")
        y_test = pd.read_csv(output_dir / "y_test.csv")
        df_train = pd.read_csv(output_dir / "df_train.csv")
        df_test = pd.read_csv(output_dir / "df_test.csv")
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(df_train) > 0
        assert len(df_test) > 0

    def test_output_files_consistent_shapes(self, runner, tmp_path, sample_raw_data):
        """Test that output files have consistent shapes."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        X_train = pd.read_csv(output_dir / "X_train_scaled.csv")
        X_test = pd.read_csv(output_dir / "X_test_scaled.csv")
        y_train = pd.read_csv(output_dir / "y_train.csv")
        y_test = pd.read_csv(output_dir / "y_test.csv")
        df_train = pd.read_csv(output_dir / "df_train.csv")
        df_test = pd.read_csv(output_dir / "df_test.csv")
        
        # Check consistent shapes
        assert len(X_train) == len(y_train) == len(df_train)
        assert len(X_test) == len(y_test) == len(df_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_target_encoded_correctly_in_output(self, runner, tmp_path, sample_raw_data):
        """Test that target is correctly encoded in output files."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        y_train = pd.read_csv(output_dir / "y_train.csv")
        y_test = pd.read_csv(output_dir / "y_test.csv")
        df_train = pd.read_csv(output_dir / "df_train.csv")
        
        assert set(y_train['Loan_Status'].unique()).issubset({0, 1})
        assert set(y_test['Loan_Status'].unique()).issubset({0, 1})
        assert set(df_train['Loan_Status'].unique()).issubset({0, 1})


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_dataframe_error(self, runner, tmp_path):
        """Test error handling with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Gender', 'Married', 'Loan_Status'])
        input_file = tmp_path / "input.csv"
        empty_df.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        result = runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        assert result.exit_code != 0

    def test_single_row_data(self, runner, tmp_path):
        """Test handling of single row data."""
        single_row = pd.DataFrame({
            'Customer_ID': ['CUST0001'],
            'Gender': ['Male'],
            'Married': ['Yes'],
            'Dependents': [0],
            'Education': ['Graduate'],
            'Self_Employed': ['No'],
            'Applicant_Income': [5000],
            'Coapplicant_Income': [0.0],
            'Loan_Amount': [100],
            'Loan_Amount_Term': [360],
            'Credit_History': [1],
            'Property_Area': ['Urban'],
            'Loan_Status': ['Y']
        })
        
        input_file = tmp_path / "input.csv"
        single_row.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        result = runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir),
            '--test-size', '0.5'
        ])
        
    def test_all_missing_numerical_column(self, runner, tmp_path):
        """Test handling when all values in a numerical column are missing."""
        data = pd.DataFrame({
            'Customer_ID': ['CUST0001', 'CUST0002'],
            'Gender': ['Male', 'Female'],
            'Married': ['Yes', 'No'],
            'Dependents': [0, 1],
            'Education': ['Graduate', 'Not Graduate'],
            'Self_Employed': ['No', 'Yes'],
            'Applicant_Income': [np.nan, np.nan],  # All missing
            'Coapplicant_Income': [0.0, 0.0],
            'Loan_Amount': [100, 200],
            'Loan_Amount_Term': [360, 180],
            'Credit_History': [1, 1],
            'Property_Area': ['Urban', 'Rural'],
            'Loan_Status': ['Y', 'N']
        })
        
        input_file = tmp_path / "input.csv"
        data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        result = runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir)
        ])
        
        assert result.exit_code == 0

    def test_large_test_size(self, runner, tmp_path, sample_raw_data):
        """Test handling of large test_size parameter."""
        input_file = tmp_path / "input.csv"
        sample_raw_data.to_csv(input_file, index=False)
        
        output_dir = tmp_path / "output"
        
        result = runner.invoke(main, [
            '--input-path', str(input_file),
            '--output-dir', str(output_dir),
            '--test-size', '0.9'
        ])
        
        assert result.exit_code == 0

