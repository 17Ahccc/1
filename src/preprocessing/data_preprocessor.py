"""
Data Preprocessing Module for Dancing with the Stars Analysis

This module handles:
- Loading data
- Missing value imputation
- Feature encoding
- Feature normalization
- Target variable creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Preprocesses the Dancing with the Stars dataset."""
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load the dataset from CSV."""
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data)} records from {self.data_path}")
        return self.raw_data
    
    def handle_missing_values(self, df):
        """
        Handle missing values in judge scores using mean imputation.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        df = df.copy()
        
        # Identify score columns
        score_cols = [col for col in df.columns if 'judge' in col and 'score' in col]
        
        # Replace 'N/A' strings with NaN
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Impute missing values with median (more robust to outliers)
        imputer = SimpleImputer(strategy='median')
        df[score_cols] = imputer.fit_transform(df[score_cols])
        
        print(f"Imputed missing values in {len(score_cols)} score columns")
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        df = df.copy()
        
        # Categorical columns to encode
        categorical_cols = [
            'celebrity_industry',
            'ballroom_partner',
            'celebrity_homestate',
            'celebrity_homecountry/region'
        ]
        
        # Handle missing values in categorical columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, 
                           drop_first=True)
        
        # Label encode season (ordinal relationship)
        if 'season' in df.columns:
            df['season'] = df['season'].astype(int)
        
        print(f"Encoded categorical features. New shape: {df.shape}")
        return df
    
    def normalize_numerical_features(self, df, fit=True):
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler or use existing fit
            
        Returns:
            pd.DataFrame: Dataframe with normalized features
        """
        df = df.copy()
        
        # Identify numerical columns to normalize
        numerical_cols = [
            'celebrity_age_during_season',
            'judge_score_mean',
            'judge_score_std',
            'hist_score_mean',
            'hist_score_std'
        ]
        
        # Only normalize columns that exist
        cols_to_normalize = [col for col in numerical_cols if col in df.columns]
        
        if cols_to_normalize:
            if fit:
                df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
            else:
                df[cols_to_normalize] = self.scaler.transform(df[cols_to_normalize])
            
            print(f"Normalized {len(cols_to_normalize)} numerical features")
        
        return df
    
    def create_target_variables(self, df):
        """
        Create target variables for modeling.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with target variables
        """
        df = df.copy()
        
        # Target variable: placement (continuous)
        # Already exists in the data
        
        # Binary target: is_eliminated (1 if eliminated, 0 if placed)
        df['is_eliminated'] = df['results'].str.contains('Eliminated', case=False, na=False).astype(int)
        
        print(f"Created target variables. Elimination rate: {df['is_eliminated'].mean():.2%}")
        return df
    
    def preprocess(self, fit=True):
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            fit (bool): Whether to fit scalers/encoders or use existing
            
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        if self.raw_data is None:
            self.load_data()
        
        print("\n=== Starting Preprocessing Pipeline ===")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(self.raw_data)
        
        # Step 2: Create target variables
        df = self.create_target_variables(df)
        
        # Step 3: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 4: Normalize numerical features (after feature engineering)
        # This will be done after feature engineering in the main pipeline
        
        self.processed_data = df
        
        print("=== Preprocessing Complete ===\n")
        return self.processed_data
    
    def save_processed_data(self, output_path):
        """
        Save processed data to CSV.
        
        Args:
            output_path (str): Path to save the processed data
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Saved processed data to {output_path}")
        else:
            print("No processed data to save. Run preprocess() first.")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("../../2026_MCM_Problem_C_Data.csv")
    processed_df = preprocessor.preprocess()
    print(f"\nFinal shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns[:20])}...")
