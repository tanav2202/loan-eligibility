## **Loan Eligibility Prediction**

This project investigates whether a loan applicant is likely to be eligible for a loan based on the information they provide. It includes data exploration, model development, documentation, and an environment setup that allows others to fully reproduce the analysis.

## **Project Summary**

At a high level, this project provides a complete workflow for building a loan eligibility classifier. The work begins with loading the main dataset located in the data folder. From there, exploratory analysis is performed to understand trends and relationships within the data. A logistic regression model is then trained to predict eligibility and the final trained model is saved in the models folder. Supporting scripts for EDA, data fetching, and model training are located in the scripts directory, while two Jupyter notebooks document the main exploratory and modelling workflow. This repository also includes documentation files, licensing information, and a reproducible environment file.

## **How to Run the Data Analysis**

Setting up and activating the environment:

``` bash
conda env create -f environment.yml
conda activate loan_env
```

Running the notebooks or Python scripts:

``` bash
jupyter notebook
python scripts/eda.py
python scripts/train_model.py
```

## **Dependencies Needed to Run the Analysis**

Installing the dependencies from the environment.yml file:

``` bash
conda env create -f environment.yml
conda activate loan_env
```

## **Licenses**

The repository contains two types of licenses. All code used in the project is covered under the MIT License, which allows others to use, modify, and distribute the code with proper attribution. All reports and documentation, including this README, fall under the CC0 1.0 Universal license, which places the written documentation in the public domain. Full licensing details are located in the LICENSE file.

## **Contributors / Authors**

The project was created by Ali Boloor, Gurleen Kaur, Justin Mak, and Tanav Singh Bajaj. Each contributor worked on different parts of the repository through their individual feature branches, including data exploration, commentary, model training, and project documentation.
