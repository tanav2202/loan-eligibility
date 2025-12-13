"""
Tests for train_model.py

Run with: pytest tests/test_train_model.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_model import main


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_train_test_data(tmp_path):
    """Create sample train/test datasets."""
    np.random.seed(42)
    n_train, n_test = 100, 30
    n_features = 10
    
    # Training data
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.DataFrame({'Loan_Status': np.random.randint(0, 2, n_train)})
    
    # Test data
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_test = pd.DataFrame({'Loan_Status': np.random.randint(0, 2, n_test)})
    
    # Save to files
    train_features = tmp_path / "X_train.csv"
    train_labels = tmp_path / "y_train.csv"
    test_features = tmp_path / "X_test.csv"
    test_labels = tmp_path / "y_test.csv"
    
    X_train.to_csv(train_features, index=False)
    y_train.to_csv(train_labels, index=False)
    X_test.to_csv(test_features, index=False)
    y_test.to_csv(test_labels, index=False)
    
    return {
        'train_features': str(train_features),
        'train_labels': str(train_labels),
        'test_features': str(test_features),
        'test_labels': str(test_labels),
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


class TestTrainModelCLI:
    """Test suite for train_model CLI."""

    def test_cli_basic_execution(self, runner, tmp_path, sample_train_test_data):
        """Test that CLI runs successfully with valid inputs."""
        output_dir = tmp_path / "results"
        
        result = runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert "Model training and evaluation complete" in result.output


    def test_cli_creates_output_directories(self, runner, tmp_path, sample_train_test_data):
        """Test that all output directories are created."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        assert (output_dir / "models").exists()
        assert (output_dir / "tables").exists()
        assert (output_dir / "figures").exists()


    def test_cli_saves_all_output_files(self, runner, tmp_path, sample_train_test_data):
        """Test that all expected output files are created."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        # Check model file
        assert (output_dir / "models" / "trained_model.pkl").exists()
        
        # Check table files
        assert (output_dir / "tables" / "cross_validation_results.csv").exists()
        assert (output_dir / "tables" / "test_scores.csv").exists()
        assert (output_dir / "tables" / "confusion_matrix.csv").exists()
        assert (output_dir / "tables" / "classification_report.csv").exists()
        
        # Check figure files
        assert (output_dir / "figures" / "roc_curve.png").exists()
        assert (output_dir / "figures" / "precision_recall_curve.png").exists()


    def test_cli_missing_train_features_error(self, runner, tmp_path, sample_train_test_data):
        """Test error when training features file is missing."""
        result = runner.invoke(main, [
            '--train-features', 'nonexistent.csv',
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(tmp_path)
        ])
        
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)


    def test_cli_missing_test_labels_error(self, runner, tmp_path, sample_train_test_data):
        """Test error when test labels file is missing."""
        result = runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', 'nonexistent.csv',
            '--output-dir', str(tmp_path)
        ])
        
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_cli_custom_random_state(self, runner, tmp_path, sample_train_test_data):
        """Test that custom random state is accepted."""
        output_dir = tmp_path / "results"
        
        result = runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir),
            '--random-state', '999'
        ])
        
        assert result.exit_code == 0


    def test_cli_custom_cv_folds(self, runner, tmp_path, sample_train_test_data):
        """Test that custom CV folds parameter works."""
        output_dir = tmp_path / "results"
        
        result = runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir),
            '--cv-folds', '5'
        ])
        
        assert result.exit_code == 0
        assert "5-fold cross-validation" in result.output


class TestModelOutputs:
    """Test suite for model outputs and saved files."""

    def test_saved_model_is_loadable(self, runner, tmp_path, sample_train_test_data):
        """Test that saved model can be loaded and used."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        model_path = output_dir / "models" / "trained_model.pkl"
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test that model can make predictions
        X_test = sample_train_test_data['X_test']
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(X_test)


    def test_cross_validation_results_format(self, runner, tmp_path, sample_train_test_data):
        """Test that cross-validation results CSV has correct format."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        cv_results = pd.read_csv(output_dir / "tables" / "cross_validation_results.csv", index_col=0)
        
        assert 'mean' in cv_results.columns
        assert 'std' in cv_results.columns
        assert 'accuracy' in cv_results.index
        assert 'precision' in cv_results.index
        assert 'recall' in cv_results.index
        assert 'f1' in cv_results.index


    def test_test_scores_format(self, runner, tmp_path, sample_train_test_data):
        """Test that test scores CSV has correct format."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        test_scores = pd.read_csv(output_dir / "tables" / "test_scores.csv")
        
        assert 'accuracy' in test_scores.columns
        assert 'F2 score (beta = 2)' in test_scores.columns
        assert len(test_scores) == 1


    def test_confusion_matrix_format(self, runner, tmp_path, sample_train_test_data):
        """Test that confusion matrix CSV has correct format."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        confusion = pd.read_csv(output_dir / "tables" / "confusion_matrix.csv", index_col=0)
        
        # Should be 2x2 for binary classification
        assert confusion.shape == (2, 2)


    def test_classification_report_format(self, runner, tmp_path, sample_train_test_data):
        """Test that classification report CSV has correct format."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        report = pd.read_csv(output_dir / "tables" / "classification_report.csv", index_col=0)
        
        assert 'precision' in report.index
        assert 'recall' in report.index
        assert 'f1-score' in report.index


    def test_plots_are_created(self, runner, tmp_path, sample_train_test_data):
        """Test that plot files are created and non-empty."""
        output_dir = tmp_path / "results"
        
        runner.invoke(main, [
            '--train-features', sample_train_test_data['train_features'],
            '--train-labels', sample_train_test_data['train_labels'],
            '--test-features', sample_train_test_data['test_features'],
            '--test-labels', sample_train_test_data['test_labels'],
            '--output-dir', str(output_dir)
        ])
        
        roc_path = output_dir / "figures" / "roc_curve.png"
        pr_path = output_dir / "figures" / "precision_recall_curve.png"
        
        # Check files exist and have content
        assert roc_path.exists()
        assert pr_path.exists()
        assert roc_path.stat().st_size > 0
        assert pr_path.stat().st_size > 0


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_perfect_predictions(self, runner, tmp_path):
        """Test model behavior with perfectly separable data."""
        # Create perfectly separable data
        X_train = pd.DataFrame({'feature': [0, 0, 1, 1] * 25})
        y_train = pd.DataFrame({'Loan_Status': [0, 0, 1, 1] * 25})
        X_test = pd.DataFrame({'feature': [0, 1] * 10})
        y_test = pd.DataFrame({'Loan_Status': [0, 1] * 10})
        
        train_features = tmp_path / "X_train.csv"
        train_labels = tmp_path / "y_train.csv"
        test_features = tmp_path / "X_test.csv"
        test_labels = tmp_path / "y_test.csv"
        output_dir = tmp_path / "results"
        
        X_train.to_csv(train_features, index=False)
        y_train.to_csv(train_labels, index=False)
        X_test.to_csv(test_features, index=False)
        y_test.to_csv(test_labels, index=False)
        
        result = runner.invoke(main, [
            '--train-features', str(train_features),
            '--train-labels', str(train_labels),
            '--test-features', str(test_features),
            '--test-labels', str(test_labels),
            '--output-dir', str(output_dir)
        ])
        
        assert result.exit_code == 0


    def test_reproducibility_with_same_random_state(self, runner, tmp_path, sample_train_test_data):
        """Test that same random state produces same results."""
        output_dir1 = tmp_path / "results1"
        output_dir2 = tmp_path / "results2"
        
        # Run twice with same random state
        for output_dir in [output_dir1, output_dir2]:
            runner.invoke(main, [
                '--train-features', sample_train_test_data['train_features'],
                '--train-labels', sample_train_test_data['train_labels'],
                '--test-features', sample_train_test_data['test_features'],
                '--test-labels', sample_train_test_data['test_labels'],
                '--output-dir', str(output_dir),
                '--random-state', '42'
            ])
        
        # Compare results
        scores1 = pd.read_csv(output_dir1 / "tables" / "test_scores.csv")
        scores2 = pd.read_csv(output_dir2 / "tables" / "test_scores.csv")
        
        pd.testing.assert_frame_equal(scores1, scores2)