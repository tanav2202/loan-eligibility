import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def preprocess_data(df, target_col='Loan_Status', test_size=0.2, random_state=42):
    """
    Preprocess loan eligibility data with one-hot encoding and split into train/test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw loan data
    target_col : str
        Name of target column (default: 'Loan_Status')
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Drop Customer_ID (not useful for prediction)
    if 'Customer_ID' in data.columns:
        data = data.drop('Customer_ID', axis=1)
    
    # Handle missing values
    # Numerical columns: fill with median
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Categorical columns: fill with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Encode target variable (Y/N to 1/0)
    y = (y == 'Y').astype(int)
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for interpretability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print(f" Data preprocessed with one-hot encoding")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Testing samples: {len(X_test_scaled)}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Target distribution - Approved: {y.sum()}, Rejected: {len(y) - y.sum()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


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
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    model_file = model_path / 'logistic_regression_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_file}")
    
    return model
