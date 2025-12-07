# Makefile for Loan Eligibility Prediction Analysis
# Author: Tanav Singh Bajaj, Ali Boloor, Gurleen Kaur, Justin Mak
# Date: 2025

.PHONY: all clean help

# Default target
all: report/loan-analysis.html

# Help target
help:
	@echo "Makefile for Loan Eligibility Prediction Analysis"
	@echo ""
	@echo "Usage:"
	@echo "  make all              Run the complete analysis pipeline"
	@echo "  make clean            Remove all generated files"
	@echo "  make help             Show this help message"
	@echo ""
	@echo "Pipeline steps:"
	@echo "  1. Download data from Kaggle"
	@echo "  2. Process and split data"
	@echo "  3. Generate EDA visualizations"
	@echo "  4. Train model and evaluate"
	@echo "  5. Render Quarto report"

# Step 1: Download data from Kaggle
data/raw/Loan\ Eligibility\ Prediction.csv:
	python scripts/download_data.py \
		--dataset-id avineshprabhakaran/loan-eligibility-prediction \
		--output-path "data/raw/Loan Eligibility Prediction.csv"

# Step 2: Process data (clean, split, validate)
data/processed/df_train.csv data/processed/X_train_scaled.csv: data/raw/Loan\ Eligibility\ Prediction.csv
	python scripts/process_data.py \
		--input-path "data/raw/Loan Eligibility Prediction.csv" \
		--output-dir data/processed \
		--test-size 0.2 \
		--random-state 522

# Step 3: Generate EDA visualizations
results/figures/univariate.png results/figures/categorical_compare.png results/figures/density_plots.png results/figures/boxplots.png results/figures/correlation_heatmap.png: data/processed/df_train.csv
	python scripts/EDA.py \
		--train-data data/processed/df_train.csv \
		--output-dir results/figures

# Step 4: Train model and generate evaluation artifacts
results/tables/test_scores.csv results/figures/roc_curve.png results/figures/precision_recall_curve.png: data/processed/X_train_scaled.csv
	python scripts/train_model.py \
		--train-features data/processed/X_train_scaled.csv \
		--train-labels data/processed/y_train.csv \
		--test-features data/processed/X_test_scaled.csv \
		--test-labels data/processed/y_test.csv \
		--output-dir results \
		--random-state 522

# Step 5: Render Quarto report
report/loan-analysis.html: report/loan-analysis.qmd report/references.bib results/tables/test_scores.csv results/figures/univariate.png
	cd report && quarto render loan-analysis.qmd

# Clean all generated files
clean:
	rm -rf data/processed/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf results/models/*
	rm -f report/loan-analysis.html
	rm -f report/loan-analysis.pdf
	rm -rf report/loan-analysis_files
	@echo "Cleaned all generated files"