"""
Download Kaggle dataset for loan eligibility prediction.

This script downloads the loan eligibility dataset from Kaggle and saves it
to the specified location. Requires Kaggle API credentials (~/.kaggle/kaggle.json).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset-id DATASET --output-path PATH
"""

import os
import click
import kaggle
import shutil


@click.command()
@click.option(
    "--dataset-id",
    type=str,
    default="avineshprabhakaran/loan-eligibility-prediction",
    help="Kaggle dataset identifier in the format 'owner/dataset-name'"
)
@click.option(
    "--output-path",
    type=str,
    default="data/raw/Loan Eligibility Prediction.csv",
    help="Path where the downloaded dataset will be saved (including filename)"
)
def main(dataset_id, output_path):
    """
    Download a Kaggle dataset and save it to the specified location.
    
    Examples:
        # Download with default settings
        python scripts/download_data.py
        
        # Download specific dataset
        python scripts/download_data.py \\
            --dataset-id avineshprabhakaran/loan-eligibility-prediction \\
            --output-path "data/raw/Loan Eligibility Prediction.csv"
    """
    
    # Extract directory from output path
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = "."
    
    # Ensure target directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading dataset: {dataset_id}")
    print(f"Saving to: {output_path}")

    # Create temporary directory for download
    temp_dir = os.path.join(output_dir, ".temp_download")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download and unzip dataset to temp directory
        kaggle.api.dataset_download_files(
            dataset_id,
            path=temp_dir,
            unzip=True
        )
        
        # Find the downloaded CSV file
        downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        
        if not downloaded_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset")
        
        if len(downloaded_files) > 1:
            print(f"Warning: Multiple CSV files found. Using: {downloaded_files[0]}")
        
        # Move the CSV to the desired location
        source = os.path.join(temp_dir, downloaded_files[0])
        shutil.move(source, output_path)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print(f"Download complete! File saved to: {output_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Clean up temp directory on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


if __name__ == "__main__":
    main()