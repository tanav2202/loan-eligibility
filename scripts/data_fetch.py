"""
Download Kaggle dataset for loan eligibility prediction.
Usage: python scripts/data_fetch.py or import download_dataset()
"""

import kaggle


def download_dataset():
    """Download loan eligibility dataset to ./data/ directory."""
    
    dataset_name = 'avineshprabhakaran/loan-eligibility-prediction'
    
    print(f"Downloading: {dataset_name}")
    
    # download to ./data/ and unzip
    kaggle.api.dataset_download_files(
        dataset_name,
        path='data/temp.csv',
        unzip=True
    )
    
    print("Downloaded to ./data/")

