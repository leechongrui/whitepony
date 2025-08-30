# preprocess.py
"""
Data preprocessing pipeline for review and business metadata.

This module provides functions to clean, validate, and prepare datasets
for machine learning training, following separation of concerns principles.
"""

import pandas as pd
from typing import Tuple
from .ingest import read_dataset, clean_business_metadata, validate_review_data


def load_and_clean_data(review_file_path: str, 
                       business_file_path: str,
                       max_records: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean both review and business datasets.
    
    Args:
        review_file_path: Path to review JSON.gz file
        business_file_path: Path to business metadata JSON.gz file
        max_records: Optional limit on records to load (for testing)
        
    Returns:
        Tuple of (cleaned_review_df, cleaned_business_df)
    """
    print("Loading review data...")
    review_df = read_dataset(review_file_path, max_records)
    review_df = validate_review_data(review_df)
    
    print("\nLoading business metadata...")
    business_df = read_dataset(business_file_path, max_records)
    business_df = clean_business_metadata(business_df)
    
    return review_df, business_df


def create_training_ready_dataset(review_df: pd.DataFrame,
                                 business_df: pd.DataFrame,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create training-ready datasets with train/val/test splits.
    
    Args:
        review_df: Cleaned review DataFrame
        business_df: Cleaned business DataFrame
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from .ingest import prepare_training_dataset, split_dataset
    
    # Merge datasets
    merged_df = prepare_training_dataset(review_df, business_df)
    
    # Split into train/val/test
    train_df, val_df, test_df = split_dataset(
        merged_df, train_ratio, val_ratio, test_ratio, random_state
    )
    
    return train_df, val_df, test_df


def get_data_summary(df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """
    Print comprehensive summary of dataset.
    
    Args:
        df: DataFrame to summarize
        dataset_name: Name of the dataset for display
    """
    print(f"\n=== {dataset_name} Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values")
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())


if __name__ == "__main__":
    # Example pipeline execution
    import os
    
    # Define file paths
    data_dir = "data/raw"
    review_file = os.path.join(data_dir, "review-California_10.json.gz")
    business_file = os.path.join(data_dir, "meta-California.json.gz")
    
    # Check if files exist
    if os.path.exists(review_file) and os.path.exists(business_file):
        # Load and clean data
        review_df, business_df = load_and_clean_data(
            review_file, business_file, max_records=1000  # Limit for testing
        )
        
        # Create training datasets
        train_df, val_df, test_df = create_training_ready_dataset(
            review_df, business_df
        )
        
        # Print summaries
        get_data_summary(train_df, "Training Set")
        get_data_summary(val_df, "Validation Set") 
        get_data_summary(test_df, "Test Set")
        
    else:
        print(f"Data files not found at {review_file} or {business_file}")
        print("Please ensure the data files are in the correct location.")