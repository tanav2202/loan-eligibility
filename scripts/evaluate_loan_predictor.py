# evaluate_loan_predictor.py

# author: [Your Name]
# date: 2024

import click
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from sklearn import set_config  # pyright: ignore[reportMissingImports]
from sklearn.metrics import fbeta_score, make_scorer, PrecisionRecallDisplay, RocCurveDisplay, classification_report, ConfusionMatrixDisplay  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import cross_validate  # pyright: ignore[reportMissingImports]

@click.command()
@click.option('--scaled-test-data', type=str, help="Path to scaled test features (X_test_scaled.csv)")
@click.option('--test-target', type=str, help="Path to test target (y_test.csv)")
@click.option('--scaled-train-data', type=str, help="Path to scaled training features (X_train_scaled.csv)")
@click.option('--train-target', type=str, help="Path to training target (y_train.csv)")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the results will be written to")
@click.option('--cv-folds', type=int, help="Number of cross-validation folds", default=10)
@click.option('--seed', type=int, help="Random seed", default=123)
def main(scaled_test_data, test_target, scaled_train_data, train_target, pipeline_from, results_to, cv_folds, seed):
    
    np.random.seed(seed)
    set_config(transform_output="pandas")
    
    # read in data & model
    # Use provided paths or default to hardcoded paths
    test_data_path = scaled_test_data or "data/processed/X_test_scaled.csv"
    test_target_path = test_target or "data/processed/y_test.csv"
    train_data_path = scaled_train_data or "data/processed/X_train_scaled.csv"
    train_target_path = train_target or "data/processed/y_train.csv"
    model_dir = pipeline_from or "results/models"
    results_dir = results_to or "results/tables"
    figures_dir = results_to or "results/figures"
    
    loan_test_X = pd.read_csv(test_data_path)
    loan_test_y = pd.read_csv(test_target_path).iloc[:, 0]  # convert to Series
    loan_train_X = pd.read_csv(train_data_path)
    loan_train_y = pd.read_csv(train_target_path).iloc[:, 0]  # convert to Series
    
    # Load the trained model
    model_path = os.path.join(model_dir, "trained_model.pkl")
    with open(model_path, 'rb') as f:
        loan_fit = pickle.load(f)
    
    # Compute accuracy
    accuracy = loan_fit.score(loan_test_X, loan_test_y)
    
    # Compute F2 score (beta = 2) - emphasizes recall
    # For loan eligibility, we want to catch eligible loans (pos_label=1)
    loan_preds = loan_test_y.to_frame().assign(
        predicted=loan_fit.predict(loan_test_X)
    )
    
    f2_beta_2_score = fbeta_score(
        loan_preds.iloc[:, 0],  # actual target
        loan_preds['predicted'],
        beta=2,
        pos_label=1  # 1 = approved loan
    )
    
    # Perform cross-validation on training data
    # Create F2 scorer (beta=2, pos_label=1)
    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
    
    # Cross-validation with multiple metrics
    cv_scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
    }
    
    cv_results = cross_validate(
        loan_fit, 
        loan_train_X, 
        loan_train_y, 
        cv=cv_folds,
        scoring=cv_scoring,
        return_train_score=False
    )
    
    # Create cross-validation results dataframe with mean and std
    cv_summary = pd.DataFrame({
        'mean': [cv_results[f'test_{metric}'].mean() for metric in cv_scoring.keys()],
        'std': [cv_results[f'test_{metric}'].std() for metric in cv_scoring.keys()]
    }, index=list(cv_scoring.keys()))
    
    # Create test scores dataframe
    test_scores = pd.DataFrame({
        'accuracy': [accuracy]
    })
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save test scores
    test_scores.to_csv(os.path.join(results_dir, "test_scores.csv"), index=False)
    
    # Save cross-validation results
    cv_summary.to_csv(os.path.join(results_dir, "cross_validation_results.csv"))
    
    # Create and save confusion matrix
    confusion_matrix = ConfusionMatrixDisplay.from_estimator(
        loan_fit,
        loan_test_X,
        loan_test_y
    )
    confusion_matrix.ax_.set_title("Confusion Matrix")
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
    plt.close()
    
    confusion_matrix = pd.crosstab(
        loan_preds.iloc[:, 0],  # actual
        loan_preds['predicted'],  # predicted
        rownames=['Actual'],
        colnames=['Predicted']
    )
    
    confusion_matrix.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))
        
    # Create and save Precision-Recall curve
    disp = PrecisionRecallDisplay.from_estimator(
        loan_fit,
        loan_test_X,
        loan_test_y
    )
    disp.ax_.set_title("Precision-Recall curve")
    plt.savefig(os.path.join(figures_dir, "precision_recall_curve.png"))
    plt.close()
    
    # Create and save ROC curve
    roc_disp = RocCurveDisplay.from_estimator(
        loan_fit,
        loan_test_X,
        loan_test_y
    )
    roc_disp.ax_.set_title("ROC curve")
    plt.savefig(os.path.join(figures_dir, "roc_curve.png"))
    plt.close()

    # Create and save classification report
    report_dict = classification_report(loan_test_y, loan_preds['predicted'], output_dict=True)
    classification_report_df = pd.DataFrame(report_dict)
    classification_report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))

if __name__ == '__main__':
    main()

