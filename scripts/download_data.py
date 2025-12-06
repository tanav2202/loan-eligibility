import os
import kaggle
import click

@click.command()
@click.option(
    "--dataset-name",
    type=str,
    required=True, 
    help="Kaggle dataset name in the format 'owner/dataset'"
)
@click.option(
    "--write-to",
    type=str,
    default="data/raw_data",
    help="Directory where the dataset will be downloaded"
)
def main(dataset_name, write_to):
    """
    Download a Kaggle dataset and save it to the specified directory.
    """

    # Ensure target directory exists
    os.makedirs(write_to, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    print(f"Saving to: {write_to}")

    kaggle.api.dataset_download_files(
        dataset_name,
        path=write_to,
        unzip=True
    )

    print(f"Download complete! Files extracted to: {write_to}")


if __name__ == "__main__":
    main()

# usage: 
# python scripts/download_data.py \
#     --dataset-name avineshprabhakaran/loan-eligibility-prediction \
#     --write-to data/raw