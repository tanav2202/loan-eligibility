# Loan Eligibility Prediction

A machine learning project that predicts loan eligibility based on applicant information using logistic regression. This repository provides a complete, reproducible workflow from data exploration to model deployment.

## Project Overview

This project implements an end-to-end pipeline for building a loan eligibility classifier:

1. **Data Ingestion**: Load and fetch datasets from Kaggle using the API
2. **Exploratory Data Analysis**: Analyze trends, distributions, and relationships in applicant data
3. **Model Training**: Build and train a logistic regression classifier
4. **Model Persistence**: Save trained models for deployment and reproducibility
5. **Model Prediction**: Generated views to check how good the trained model is

### Repository Structure
```
.
├── data/              # Raw and processed datasets
├── models/            # Trained model artifacts (.pkl files)
├── scripts/           # Modular Python scripts for data fetching and model training
├── eda.ipynb          # Exploratory data analysis notebook
├── analysis.ipynb     # Main analysis and modeling notebook
├── environment.yml    # Conda environment specification
├── CODE_OF_CONDUCT.md # Community guidelines
├── CONTRIBUTING.md    # Contribution guidelines
└── LICENSE            # Project licenses
```

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) package manager
- Jupyter Lab or Jupyter Notebook (install via `conda install jupyterlab`)
- [Kaggle account](https://www.kaggle.com) for dataset access

### Installation

1. **Clone the repository**
```bash
   git clone git@github.com:tanav2202/loan-eligibility.git
   cd loan-eligibility
```

2. **Create and activate the conda environment**
```bash
   conda env create -f environment.yml
   conda activate loan-analysis
```

3. **Configure Kaggle API**
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
   - Navigate to your account settings and create an API token
   - Download the `kaggle.json` credentials file
   - Place it in `~/.kaggle/` (create the directory if it doesn't exist)
   
   For detailed setup instructions, see the [Kaggle API documentation](https://www.kaggle.com/discussions/getting-started/524433).

### Running the Analysis

Launch Jupyter and open the analysis notebook:
```bash
jupyter lab loan-analysis.ipynb
```

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

Dataset sourced from Kaggle. Special thanks to the MDS program at UBC for project guidance and support.
