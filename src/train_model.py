"""
Train and evaluate logistic regression model for loan eligibility prediction.

This script:
1. Loads processed training and test data
2. Trains a logistic regression model
3. Performs cross-validation on training data
4. Evaluates model on test data
5. Generates evaluation plots (ROC, Precision-Recall)
6. Saves model and all results

Usage:
    python scripts/train_model.py [OPTIONS]
"""

import os
import click
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    fbeta_score,
    classification_report,
    PrecisionRecallDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


@click.command()
@click.option(
    "--train-features",
    type=str,
    required=True,
    help="Path to scaled training features CSV (e.g., X_train_scaled.csv)"
)
@click.option(
    "--train-labels",
    type=str,
    required=True,
    help="Path to training labels CSV (e.g., y_train.csv)"
)
@click.option(
    "--test-features",
    type=str,
    required=True,
    help="Path to scaled test features CSV (e.g., X_test_scaled.csv)"
)
@click.option(
    "--test-labels",
    type=str,
    required=True,
    help="Path to test labels CSV (e.g., y_test.csv)"
)
@click.option(
    "--output-dir",
    type=str,
    default="results",
    help="Base directory for saving results (default: results)"
)
@click.option(
    "--random-state",
    type=int,
    default=522,
    help="Random seed for reproducibility (default: 522)"
)
@click.option(
    "--cv-folds",
    type=int,
    default=10,
    help="Number of cross-validation folds (default: 10)"
)
def main(train_features, train_labels, test_features, test_labels, 
         output_dir, random_state, cv_folds):
    """
    Train and evaluate a logistic regression model for loan eligibility prediction.
    
    Examples:
        python scripts/train_model.py \\
            --train-features data/processed/X_train_scaled.csv \\
            --train-labels data/processed/y_train.csv \\
            --test-features data/processed/X_test_scaled.csv \\
            --test-labels data/processed/y_test.csv \\
            --output-dir results \\
            --random-state 522
    """
    
    np.random.seed(random_state)
    
    # Create output directories
    model_dir = os.path.join(output_dir, "models")
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    
    for directory in [model_dir, tables_dir, figures_dir]:
        os.makedirs(directory, exist_ok=True)

    # Load training data
    print("Loading training data...")
    if not os.path.exists(train_features):
        raise FileNotFoundError(f"Training features not found: {train_features}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Training labels not found: {train_labels}")
    
    X_train = pd.read_csv(train_features)
    y_train = pd.read_csv(train_labels).iloc[:, 0]
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Load test data
    print("\nLoading test data...")
    if not os.path.exists(test_features):
        raise FileNotFoundError(f"Test features not found: {test_features}")
    if not os.path.exists(test_labels):
        raise FileNotFoundError(f"Test labels not found: {test_labels}")
    
    X_test = pd.read_csv(test_features)
    y_test = pd.read_csv(test_labels).iloc[:, 0]
    
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Train the model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    
    train_accuracy = model.score(X_train, y_train)
    print(f"  Training accuracy: {train_accuracy:.4f}")

    # Cross-validation on training data
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
    }
    
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=cv_folds,
        scoring=cv_scoring,
        return_train_score=False
    )
    
    # Create cross-validation results dataframe
    cv_summary = pd.DataFrame({
        'mean': [cv_results[f'test_{metric}'].mean() for metric in cv_scoring.keys()],
        'std': [cv_results[f'test_{metric}'].std() for metric in cv_scoring.keys()]
    }, index=list(cv_scoring.keys()))
    
    print("  Cross-validation results:")
    print(cv_summary)
    
    # Save cross-validation results
    cv_path = os.path.join(tables_dir, "cross_validation_results.csv")
    cv_summary.to_csv(cv_path)
    print(f"   Saved: {cv_path}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    # Calculate F2 score (emphasizes recall)
    f2_score = fbeta_score(y_test, y_pred, beta=2, pos_label=1)
    
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  F2 score (beta=2): {f2_score:.4f}")
    
    # Save test scores
    test_scores = pd.DataFrame({
        'accuracy': [test_accuracy], 
        'F2 score (beta = 2)': [f2_score]
    })
    test_scores_path = os.path.join(tables_dir, "test_scores.csv")
    test_scores.to_csv(test_scores_path, index=False)
    print(f"   Saved: {test_scores_path}")

    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        y_test,
        y_pred,
        rownames=['Actual'],
        colnames=['Predicted']
    )
    confusion_path = os.path.join(tables_dir, "confusion_matrix.csv")
    confusion_matrix.to_csv(confusion_path)
    print(f"   Saved: {confusion_path}")

    # Create classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(report_dict)
    report_path = os.path.join(tables_dir, "classification_report.csv")
    classification_report_df.to_csv(report_path)
    print(f"   Saved: {report_path}")

    # Generate evaluation plots
    print("\nGenerating evaluation plots...")
    
    # ROC curve
    roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    roc_disp.ax_.set_title("ROC Curve")
    roc_path = os.path.join(figures_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {roc_path}")

    # Precision-Recall curve
    pr_disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    pr_disp.ax_.set_title("Precision-Recall Curve")
    pr_path = os.path.join(figures_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {pr_path}")

    # Save trained model
    model_path = os.path.join(model_dir, "trained_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n Saved trained model to: {model_path}")

    print("\n" + "="*60)
    print("Model training and evaluation complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  Models: {model_dir}/")
    print(f"  Tables: {tables_dir}/")
    print(f"  Figures: {figures_dir}/")


if __name__ == "__main__":
    main()