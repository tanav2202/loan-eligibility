

import os
import click
import pickle


import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def train_classifier(X_train, y_train, model_dir='models', max_iter=1000, random_state=42):
    """
    Train a Logistic Regression classifier and save to disk.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    model_dir : str
        Directory to save trained model (default: 'models')
    max_iter : int
        Maximum iterations for solver (default: 1000)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    LogisticRegression
        Trained model
    """
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Calculate training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    print(f"\nModel trained successfully")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    
    # Save model to disk
    # model_path = Path(model_dir)
    # model_path.mkdir(parents=True, exist_ok=True)
    
    # model_file = model_path / 'logistic_regression_model.pkl'
    # with open(model_file, 'wb') as f:
    #     pickle.dump(model, f)
    
    # print(f"\nModel saved to: {model_file}")
    
    return model


@click.command()
@click.option(
    "--x-train",
    type=str,
    required=True,
    help="Path to X_train.csv (processed training features)"
)
@click.option(
    "--y-train",
    type=str,
    required=True,
    help="Path to y_train.csv (processed training target)"
)
@click.option(
    "--model-to",
    type=str,
    required=True,
    help="Directory where the trained model will be saved"
)
def main(x_train, y_train, model_to):
    """
    Load X_train and y_train, train the classifier, and save the model.
    """

    # ---------------------------
    # 1. Load training data
    # ---------------------------
    print("Loading training data...")

    if not os.path.exists(x_train):
        raise FileNotFoundError(f"X_train file not found: {x_train}")
    if not os.path.exists(y_train):
        raise FileNotFoundError(f"y_train file not found: {y_train}")

    X_train_df = pd.read_csv(x_train)
    y_train_df = pd.read_csv(y_train).iloc[:, 0]  # convert to Series

    print(f"  X_train shape: {X_train_df.shape}")
    print(f"  y_train shape: {y_train_df.shape}")

    # ---------------------------
    # 2. Train the model
    # ---------------------------
    print("Training model...")

    model = train_classifier(
        X_train=X_train_df,
        y_train=y_train_df
    )

    print("Model training complete.")

    # ---------------------------
    # 3. Save trained model
    # ---------------------------
    os.makedirs(model_to, exist_ok=True)
    model_path = os.path.join(model_to, "trained_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved trained model to: {model_path}")


if __name__ == "__main__":
    main()


# usage
# python scripts/train_model.py \
#     --x-train data/processed/X_train_scaled.csv \
#     --y-train data/processed/y_train.csv \
#     --model-to results/models