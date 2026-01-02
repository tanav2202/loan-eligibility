"""
Tests for download_data.py

Run with: pytest tests/test_download_data.py -v
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.download_data import main


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_kaggle_api():
    """Mock the Kaggle API to avoid actual downloads."""
    with patch('scripts.download_data.kaggle.api') as mock_api:
        yield mock_api


@pytest.fixture
def setup_temp_csv(tmp_path):
    """Create a mock CSV file in temp directory."""
    def _setup(output_path):
        temp_dir = output_path.parent / ".temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "data.csv").write_text("col1,col2\n1,2\n")
        return temp_dir
    return _setup


class TestDownloadData:
    """Test suite for download_data script."""

    def test_default_parameters(self, runner, mock_kaggle_api, tmp_path, setup_temp_csv):
        """Test script runs with default parameters."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            setup_temp_csv(Path("data/raw/Loan Eligibility Prediction.csv"))
            mock_kaggle_api.dataset_download_files.return_value = None
            
            result = runner.invoke(main)
            
            assert result.exit_code == 0
            assert "avineshprabhakaran/loan-eligibility-prediction" in result.output


    def test_custom_parameters(self, runner, mock_kaggle_api, tmp_path, setup_temp_csv):
        """Test script with custom dataset ID and output path."""
        output_path = tmp_path / "custom" / "data.csv"
        
        with runner.isolated_filesystem(temp_dir=tmp_path):
            setup_temp_csv(output_path)
            mock_kaggle_api.dataset_download_files.return_value = None
            
            result = runner.invoke(main, [
                '--dataset-id', 'custom/dataset',
                '--output-path', str(output_path)
            ])
            
            assert result.exit_code == 0
            assert "custom/dataset" in result.output


    def test_creates_nested_directories(self, runner, mock_kaggle_api, tmp_path, setup_temp_csv):
        """Test that nested output directories are created."""
        output_path = tmp_path / "new" / "nested" / "dir" / "data.csv"
        
        with runner.isolated_filesystem(temp_dir=tmp_path):
            setup_temp_csv(output_path)
            mock_kaggle_api.dataset_download_files.return_value = None
            
            result = runner.invoke(main, ['--output-path', str(output_path)])
            
            assert result.exit_code == 0
            assert output_path.parent.exists()


    def test_kaggle_api_error_handling(self, runner, mock_kaggle_api):
        """Test that Kaggle API errors are handled gracefully."""
        mock_kaggle_api.dataset_download_files.side_effect = Exception("API Error")
        
        result = runner.invoke(main)
        
        assert result.exit_code != 0
        assert "Error downloading dataset" in result.output


    def test_no_csv_found_error(self, runner, mock_kaggle_api, tmp_path):
        """Test error when no CSV files are found."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            temp_dir = Path("data/raw/.temp_download")
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "data.txt").write_text("not a csv")
            
            mock_kaggle_api.dataset_download_files.return_value = None
            
            result = runner.invoke(main)
            
            assert result.exit_code != 0


    def test_multiple_csv_warning(self, runner, mock_kaggle_api, tmp_path):
        """Test warning when multiple CSV files are found."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            temp_dir = Path("data/raw/.temp_download")
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "file1.csv").write_text("data1\n")
            (temp_dir / "file2.csv").write_text("data2\n")
            
            mock_kaggle_api.dataset_download_files.return_value = None
            
            result = runner.invoke(main)
            
            assert result.exit_code == 0
            assert "Multiple CSV files found" in result.output


    def test_temp_directory_cleanup(self, runner, mock_kaggle_api, tmp_path, setup_temp_csv):
        """Test that temp directory is cleaned up after download."""
        output_path = tmp_path / "output.csv"
        
        with runner.isolated_filesystem(temp_dir=tmp_path):
            temp_dir = setup_temp_csv(output_path)
            mock_kaggle_api.dataset_download_files.return_value = None
            
            runner.invoke(main, ['--output-path', str(output_path)])
            
            assert not temp_dir.exists()


    def test_temp_cleanup_on_error(self, runner, mock_kaggle_api, tmp_path):
        """Test temp directory cleanup when errors occur."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            temp_dir = Path("data/raw/.temp_download")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            mock_kaggle_api.dataset_download_files.side_effect = Exception("Error")
            
            runner.invoke(main)
            
            assert not temp_dir.exists()