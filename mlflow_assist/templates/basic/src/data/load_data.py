"""
Data loading and preprocessing functions.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(
    data_path: str = "data/raw/data.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split data into train and test sets.
    
    Args:
        data_path: Path to the data file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split into train and test sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

