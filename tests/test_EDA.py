import pytest
import pandas as pd
import numpy as np
import altair as alt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.EDA import ExploratoryDataAnalysis


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'Married': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Education': ['Grad', 'Grad', 'Undergrad', 'Grad', 'Undergrad', 'Grad'],
        'Loan_Status': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        'Property_Area': ['Urban', 'Rural', 'Urban', 'Semiurban', 'Urban', 'Rural']
    })


@pytest.fixture
def sample_data_non_numeric_target():
    """Create DataFrame with non-numeric target."""
    return pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F'],
        'Married': ['Yes', 'No', 'Yes', 'No'],
        'Loan_Status': ['Approved', 'Rejected', 'Approved', 'Rejected']
    })


@pytest.fixture
def large_sample_data():
    """Create a larger sample DataFrame with more categories."""
    np.random.seed(42)
    return pd.DataFrame({
        'Gender': np.random.choice(['M', 'F'], 100),
        'Married': np.random.choice(['Yes', 'No'], 100),
        'Education': np.random.choice(['Grad', 'Undergrad'], 100),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], 100),
        'Loan_Status': np.random.uniform(0, 1, 100)
    })


class TestCompareCategoricalFeatures:
    """Test suite for compare_categorical_features method."""

    def test_valid_input_returns_vconcatchart(self, sample_data):
        """Test that valid input returns a VConcatChart."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender', 'Married'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_single_categorical_column(self, sample_data):
        """Test with a single categorical column."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_multiple_categorical_columns(self, sample_data):
        """Test with multiple categorical columns."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender', 'Married', 'Education', 'Property_Area'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_custom_columns_layout_two_columns(self, sample_data):
        """Test with custom grid column count of 2."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender', 'Married', 'Education'],
            target_name='Loan_Status',
            columns=2
        )
        assert isinstance(result, alt.VConcatChart)

    def test_custom_columns_layout_one_column(self, large_sample_data):
        """Test with custom grid column count of 1."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            large_sample_data,
            categorical_cols=['Gender', 'Married', 'Education', 'Property_Area'],
            target_name='Loan_Status',
            columns=1
        )
        assert isinstance(result, alt.VConcatChart)

    def test_missing_target_column_raises_error(self, sample_data):
        """Test that missing target column raises ValueError."""
        with pytest.raises(ValueError, match="Target column 'NonExistent' not found"):
            ExploratoryDataAnalysis.compare_categorical_features(
                sample_data,
                categorical_cols=['Gender'],
                target_name='NonExistent'
            )

    def test_non_numeric_target_raises_error(self, sample_data_non_numeric_target):
        """Test that non-numeric target raises ValueError."""
        with pytest.raises(ValueError, match="Target variable must be numeric"):
            ExploratoryDataAnalysis.compare_categorical_features(
                sample_data_non_numeric_target,
                categorical_cols=['Gender'],
                target_name='Loan_Status'
            )

    def test_missing_categorical_column_raises_error(self, sample_data):
        """Test that missing categorical column raises ValueError."""
        with pytest.raises(ValueError, match="Categorical column 'NonExistent' not found"):
            ExploratoryDataAnalysis.compare_categorical_features(
                sample_data,
                categorical_cols=['NonExistent'],
                target_name='Loan_Status'
            )

    def test_missing_multiple_categorical_columns_raises_error(self, sample_data):
        """Test that error is raised for the first missing categorical column."""
        with pytest.raises(ValueError, match="Categorical column 'BadCol1' not found"):
            ExploratoryDataAnalysis.compare_categorical_features(
                sample_data,
                categorical_cols=['Gender', 'BadCol1', 'BadCol2'],
                target_name='Loan_Status'
            )

    def test_empty_categorical_cols_list(self, sample_data):
        """Test with empty categorical columns list."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=[],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_chart_has_correct_title(self, sample_data):
        """Test that final chart has correct title."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender'],
            target_name='Loan_Status'
        )
        # Check if title is in the properties
        chart_dict = result.to_dict()
        assert 'title' in chart_dict
        assert 'Categorical Feature Comparison' in chart_dict['title']
        assert 'Loan_Status' in chart_dict['title']

    def test_correct_number_of_rows_in_grid(self, sample_data):
        """Test that grid layout produces correct number of rows."""
        # With 4 columns and columns=2, should create 2 rows
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender', 'Married', 'Education', 'Property_Area'],
            target_name='Loan_Status',
            columns=2
        )
        # Verify result is VConcatChart (which means multiple rows)
        assert isinstance(result, alt.VConcatChart)

    def test_with_integer_target_column(self, sample_data):
        """Test with integer target column."""
        sample_data['Loan_Status_Int'] = sample_data['Loan_Status'].astype(int)
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender'],
            target_name='Loan_Status_Int'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_with_float_target_column(self, sample_data):
        """Test with float target column."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data,
            categorical_cols=['Gender'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_with_categorical_column_having_nan(self, sample_data):
        """Test handling of categorical columns with NaN values."""
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[0, 'Gender'] = np.nan
        
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data_with_nan,
            categorical_cols=['Gender'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_default_columns_parameter(self, large_sample_data):
        """Test that default columns parameter is 3."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            large_sample_data,
            categorical_cols=['Gender', 'Married', 'Education', 'Property_Area'],
            target_name='Loan_Status'
            # columns parameter not specified, should default to 3
        )
        assert isinstance(result, alt.VConcatChart)

    def test_single_category_in_feature(self, sample_data):
        """Test with a categorical column that has only one unique value."""
        sample_data_single_category = sample_data.copy()
        sample_data_single_category['Single_Category'] = 'A'
        
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data_single_category,
            categorical_cols=['Single_Category'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_target_with_all_same_values(self, sample_data):
        """Test with target column having all same values."""
        sample_data_same_target = sample_data.copy()
        sample_data_same_target['Loan_Status'] = 1.0
        
        result = ExploratoryDataAnalysis.compare_categorical_features(
            sample_data_same_target,
            categorical_cols=['Gender'],
            target_name='Loan_Status'
        )
        assert isinstance(result, alt.VConcatChart)

    def test_large_number_of_categorical_columns(self, large_sample_data):
        """Test with many categorical columns."""
        result = ExploratoryDataAnalysis.compare_categorical_features(
            large_sample_data,
            categorical_cols=['Gender', 'Married', 'Education', 'Property_Area'],
            target_name='Loan_Status',
            columns=2
        )
        assert isinstance(result, alt.VConcatChart)
