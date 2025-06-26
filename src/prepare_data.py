"""
Data preparation script for student admission prediction
This script loads the raw data, cleans it, and splits it into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data():
    """Load the admission dataset from raw data folder"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to project root and then to data/raw
    data_path = os.path.join(script_dir, '..', 'data', 'raw', 'admission.csv')
    df = pd.read_csv(data_path)
    return df

def clean_data(df):
    """Clean the dataset and prepare features"""
    # Display basic info about the dataset
    print("Dataset shape:", df.shape)
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Remove Serial No. column as it's not useful for prediction
    if 'Serial No.' in df.columns:
        df = df.drop('Serial No.', axis=1)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Rename columns for easier handling
    column_mapping = {
        'GRE Score': 'GRE_Score',
        'TOEFL Score': 'TOEFL_Score',
        'University Rating': 'University_Rating',
        'LOR': 'LOR',
        'Chance of Admit': 'Chance_of_Admit'
    }
    df = df.rename(columns=column_mapping)
    
    print("\nCleaned dataset columns:", df.columns.tolist())
    print("Cleaned dataset shape:", df.shape)
    
    return df

def prepare_features_target(df):
    """Separate features and target variable"""
    # Define target variable
    target_col = 'Chance_of_Admit'
    
    # Features to use for modeling
    feature_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nFeatures used:", feature_cols)
    
    # Display basic statistics
    print("\nFeatures statistics:")
    print(X.describe())
    print("\nTarget statistics:")
    print(y.describe())
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    print(f"\nData split completed:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Test size: {test_size*100}%")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """Save processed data to CSV files"""
    # Create processed data directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the datasets
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print(f"\nProcessed data saved to {processed_dir}/")
    print("Files created:")
    print("- X_train.csv")
    print("- X_test.csv") 
    print("- y_train.csv")
    print("- y_test.csv")

def main():
    """Main function to execute data preparation pipeline"""
    print("=" * 50)
    print("STUDENT ADMISSION DATA PREPARATION")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data()
    
    # Clean data
    print("\n2. Cleaning data...")
    df_clean = clean_data(df)
    
    # Prepare features and target
    print("\n3. Preparing features and target...")
    X, y = prepare_features_target(df_clean)
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save processed data
    print("\n5. Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
