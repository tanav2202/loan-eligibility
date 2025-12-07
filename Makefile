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
# 5. Test model (new)
# -------------------------------------------------------
results/models/test_results.txt: \
scripts/test_model.py \
results/models/trained_model.pickle \
data/processed/X_test_scaled.csv \
data/processed/y_test.csv
	python scripts/test_model.py \
		--model results/models/trained_model.pickle \
		--x-test data/processed/X_test_scaled.csv \
		--y-test data/processed/y_test.csv \
		--write-to results/models


# -------------------------------------------------------
# 6. Build HTML report with Quarto
# -------------------------------------------------------
report/_build/html/index.html: \
reports/report.qmd \
results/models/trained_model.pickle \
results/models/test_results.txt \
results/figures/boxplots.png \
results/figures/categorical_compare.png \
results/figures/correlation_heatmap.png \
results/figures/density_plots.png \
results/figures/univariate.png
	quarto render reports/report.qmd --to html


# -------------------------------------------------------
# 7. Cleaning rule (new)
# -------------------------------------------------------
clean:
	rm -rf data/processed/*.csv
	rm -rf results/models/*
	rm -rf results/figures/*.png
	rm -rf report/_build