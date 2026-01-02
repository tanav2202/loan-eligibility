# Loan Eligibility Prediction

A machine learning project that predicts loan eligibility based on applicant information using logistic regression. This repository provides a complete, reproducible workflow from data exploration to model deployment.

## Project Overview

This project implements an end-to-end pipeline for building a loan eligibility classifier:

1. **Data Download**: Fetch dataset from Kaggle using the API
2. **Data Processing**: Clean, transform, and split data for analysis
3. **Exploratory Data Analysis**: Generate visualizations and tables to understand the data
4. **Model Training**: Build, train, and evaluate a logistic regression classifier
5. **Report Generation**: Produce a professional analysis report using Quarto

## Repository Structure
```
.
├── data/
│   ├── raw/                      # Original downloaded datasets
│   └── processed/                # Cleaned and split data
├── results/
│   ├── figures/                  # EDA and model evaluation plots
│   ├── tables/                   # Analysis tables and metrics
│   └── models/                   # Trained model artifacts
├── src/                      # Modular Python analysis scripts
│   ├── download_data.py          # Script 1: Download data from Kaggle
│   ├── process_data.py           # Script 2: Clean and split data
│   ├── EDA.py                    # Script 3: Generate EDA visualizations
│   └── train_model.py            # Script 4: Train and evaluate model
├── reports/
│   ├── loan-analysis.qmd         # Quarto report document
│   └── references.bib            # Bibliography file
├── environment.yml               # Conda environment specification
├── conda-lock.yml                # Locked dependencies for reproducibility
├── Dockerfile                    # Docker container specification
├── docker-compose.yml            # Docker compose configuration
├── Makefile                      # Automation of analysis pipeline
├── CODE_OF_CONDUCT.md            # Community guidelines
├── CONTRIBUTING.md               # Contribution guidelines
└── LICENSE                       # Project license
```

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (recommended)
- OR [Conda](https://docs.conda.io/en/latest/) package manager
- [Kaggle account](https://www.kaggle.com) for dataset access

### Installation

#### Using Docker

1. **Clone the repository**
```bash
git clone git@github.com:tanav2202/loan-eligibility.git
cd loan-eligibility
```

2. **Configure Kaggle API**
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
   - Navigate to your account settings and create an API token
   - Download the `kaggle.json` credentials file
   - Place it in `~/.kaggle/` (create the directory if it doesn't exist)
   
   For detailed setup instructions, see the [Kaggle API documentation](https://www.kaggle.com/discussions/getting-started/524433).
3. **Build and run the Docker container**
```bash
docker-compose build
docker-compose up -d
```

4. **Access the container and run analysis**
```bash
docker exec -it loan-analysis bash

make all
```

The `make all` command runs the complete analysis pipeline inside the container.

## Running the Analysis

### Automated Pipeline (Using Make)

To run the complete analysis pipeline:

```bash
make all
```

This will execute all four scripts in sequence and generate the final report.

To clean all generated files and start fresh:

```bash
make clean
```

### Manual Execution (Individual Scripts)

You can also run each script individually with command-line arguments:

#### 1. Download Data

Downloads the loan eligibility dataset from Kaggle.

```bash
python src/download_data.py \
    --dataset-id avineshprabhakaran/loan-eligibility-prediction \
    --output-path "data/raw/Loan Eligibility Prediction.csv"

```

**Arguments:**
- `--dataset-id`: Kaggle dataset identifier (default: "vikasp001/loan-eligibility-prediction")
- `--output-path`: Path where the downloaded dataset will be saved (default: "data/raw/Loan Eligibility Prediction.csv")

#### 2. Process Data

Cleans the raw data, handles missing values, and splits into train/test sets.

```bash
python src/process_data.py \
    --input-path "data/raw/Loan Eligibility Prediction.csv" \
    --output-dir data/processed \
    --test-size 0.2 \
    --random-state 522
```

**Arguments:**
- `--input-path`: Path to the raw CSV file to process
- `--output-dir`: Directory where processed datasets will be saved (default: "data/processed")
- `--test-size`: Proportion of data for test set (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 522)


#### 3. Exploratory Data Analysis

Generates visualizations and summary tables for understanding the dataset.

```bash
 python src/EDA.py \        
    --train-data data/processed/df_train.csv \
    --output-dir results/figures

```

**Arguments:**
- `--train-data`: Path to the training dataset CSV file
- `--output-dir`: Directory where EDA figures will be saved (default: "results/figures")


#### 4. Train and Evaluate Model

Trains a logistic regression model and evaluates its performance.

```bash
python src/train_model.py \
    --train-features data/processed/X_train_scaled.csv \
    --train-labels data/processed/y_train.csv \
    --test-features data/processed/X_test_scaled.csv \
    --test-labels data/processed/y_test.csv \
    --output-dir results \
    --random-state 522
```

**Arguments:**
- `--train-features`: Path to scaled training features CSV
- `--train-labels`: Path to training labels CSV
- `--test-features`: Path to scaled test features CSV
- `--test-labels`: Path to test labels CSV
- `--output-dir`: Base directory for saving results (default: "results")
- `--random-state`: Random seed for reproducibility (default: 522)

### Generating the Report

After running all scripts, generate the final Quarto report:

```bash
quarto render reports/loan-analysis.qmd
```

The rendered report will be saved as `reports/loan-analysis.html` (and `.pdf` ).

## Dependencies

All project dependencies are managed through:
- `environment.yml`: High-level conda dependencies
- `conda-lock`: Platform-specific locked dependencies for reproducibility
- `Dockerfile`: Containerized environment specification

Key dependencies include:
- Python 3.11
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning
- matplotlib, seaborn: Visualization
- click: Command-line interface
- kaggle: Data download
- quarto: Report generation

## Development

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

### Git Workflow

- Create feature branches for new work
- Submit pull requests for review
- At least one team member must review before merging
- Use meaningful commit messages
- Communicate via GitHub issues

## License

This project uses a dual-license structure:

- **Code**: Licensed under the [MIT License](LICENSE) - free to use, modify, and distribute with attribution
- **Documentation**: Licensed under [CC0 1.0 Universal](LICENSE) - public domain dedication for all written materials and reports

See the [LICENSE](LICENSE) file for complete details.

## Contributors

This project was developed collaboratively by:

- **Tanav Singh Bajaj**
- **Ali Boloor**
- **Gurleen Kaur**
- **Justin Mak**

Each team member contributed through feature branches covering data exploration, model development, documentation, and project infrastructure.

## Acknowledgments

- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/avineshprabhakaran/loan-eligibility-prediction)
- Special thanks to the MDS program at UBC for project guidance and support
- Example project structure inspired by the [Breast Cancer Predictor](https://github.com/ttimbers/breast_cancer_predictor_py)