# -------------------------------------------------------
# Build everything
# -------------------------------------------------------
all: report/_build/html/index.html


# -------------------------------------------------------
# 1. Download raw data
# -------------------------------------------------------
data/raw/Loan\ Eligibility\ Prediction.csv: scripts/download_data.py
	python scripts/download_data.py \
		--dataset-name avineshprabhakaran/loan-eligibility-prediction \
		--write-to data/raw


# -------------------------------------------------------
# 2. Process data → produce train/test CSV files
# -------------------------------------------------------
data/processed/df_test.csv \
data/processed/df_train.csv \
data/processed/X_test_scaled.csv \
data/processed/X_train_scaled.csv \
data/processed/y_test.csv \
data/processed/y_train.csv: \
scripts/process_data.py \
data/raw/Loan\ Eligibility\ Prediction.csv
	python scripts/process_data.py \
		--raw-data "data/raw/Loan Eligibility Prediction.csv" \
		--write-to data/processed


# -------------------------------------------------------
# 3. Perform EDA → save plots
# -------------------------------------------------------
results/figures/boxplots.png \
results/figures/categorical_compare.png \
results/figures/correlation_heatmap.png \
results/figures/density_plots.png \
results/figures/univariate.png: \
scripts/eda.py data/processed/df_train.csv
	python scripts/eda.py \
		--processed-training-data data/processed/df_train.csv \
		--plot-to results/figures


# -------------------------------------------------------
# 4. Train model
# -------------------------------------------------------
results/models/trained_model.pickle: \
scripts/train_model.py \
data/processed/X_train_scaled.csv \
data/processed/y_train.csv
	python scripts/train_model.py \
		--x-train data/processed/X_train_scaled.csv \
		--y-train data/processed/y_train.csv \
		--model-to results/models


# -------------------------------------------------------
# 5. Evaluate and test model 
# -------------------------------------------------------
results/tables/test_scores.csv \
results/tables/cross_validation_results.csv \
results/tables/confusion_matrix.csv \
results/tables/classification_report.csv \
results/figures/precision_recall_curve.png \
results/figures/roc_curve.png : \
scripts/evaluate_loan_predictor.py \
data/processed/X_test_scaled.csv \
data/processed/y_test.csv \
data/processed/X_train_scaled.csv \
data/processed/y_train.csv \
results/models/trained_model.pkl
	python scripts/evaluate_loan_predictor.py \
		--scaled-test-data data/processed/X_test_scaled.csv \
		--test-target data/processed/y_test.csv \
		--scaled-train-data data/processed/X_train_scaled.csv \
		--train-target data/processed/y_train.csv \
		--pipeline-from results/models \
		--results-to results/

# -------------------------------------------------------
# 6. Build HTML report with Quarto
# -------------------------------------------------------
report/_build/html/index.html: \
reports/report.qmd \
results/models/trained_model.pkl \
results/tables/test_scores.csv \
results/tables/cross_validation_results.csv \
results/tables/confusion_matrix.csv \
results/tables/classification_report.csv \
results/figures/boxplots.png \
results/figures/categorical_compare.png \
results/figures/correlation_heatmap.png \
results/figures/density_plots.png \
results/figures/univariate.png
	quarto render reports/report.qmd --to html


# -------------------------------------------------------
# 7. Cleaning rule 
# -------------------------------------------------------
clean:
	rm -rf data/processed/*.csv
	rm -rf results/models/*
	rm -rf results/figures/*.png
	rm -rf results/tables/*.csv
	rm -rf report/_build