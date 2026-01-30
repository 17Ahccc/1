"""
Data preprocessing module for MCM 2026 Problem C
Handles loading, cleaning, and encoding of Dancing with the Stars data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, List


class DataPreprocessor:
    """
    Preprocesses the Dancing with the Stars dataset according to specifications:
    - Handles missing values (fills with 0)
    - Encodes categorical variables
    - Standardizes numerical features
    """
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with data file path
        
        Args:
            data_path: Path to CSV data file
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame with loaded data
        """
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} records from {self.data_path}")
        return self.data
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values according to assumptions:
        - Judge scores: Fill with 0 (as per Assumption 1)
        - N/A values: Convert to 0
        
        Returns:
            DataFrame with missing values handled
        """
        # Replace 'N/A' strings with NaN
        self.data = self.data.replace('N/A', np.nan)
        
        # Get all judge score columns
        judge_score_cols = [col for col in self.data.columns if 'judge' in col and 'score' in col]
        
        # Fill missing judge scores with 0
        for col in judge_score_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col] = self.data[col].fillna(0)
        
        print(f"Handled missing values in {len(judge_score_cols)} judge score columns")
        return self.data
    
    def encode_categorical_variables(self) -> pd.DataFrame:
        """
        Encode categorical variables:
        - celebrity_industry: One-hot encoding
        - celebrity_name, ballroom_partner: Label encoding
        - celebrity_homestate, celebrity_homecountry/region: Label encoding
        
        Returns:
            DataFrame with encoded categorical variables
        """
        # One-hot encode industry
        if 'celebrity_industry' in self.data.columns:
            industry_dummies = pd.get_dummies(self.data['celebrity_industry'], 
                                             prefix='industry')
            self.data = pd.concat([self.data, industry_dummies], axis=1)
            print(f"One-hot encoded celebrity_industry into {len(industry_dummies.columns)} columns")
        
        # Label encode other categorical variables
        categorical_cols = ['celebrity_name', 'ballroom_partner', 
                          'celebrity_homestate', 'celebrity_homecountry/region']
        
        for col in categorical_cols:
            if col in self.data.columns:
                # Handle missing values in categorical columns
                self.data[col] = self.data[col].fillna('Unknown')
                
                # Create and fit encoder
                encoder = LabelEncoder()
                self.data[f'{col}_encoded'] = encoder.fit_transform(self.data[col])
                self.encoders[col] = encoder
                print(f"Label encoded {col}")
        
        return self.data
    
    def standardize_numerical_features(self) -> pd.DataFrame:
        """
        Standardize numerical features:
        - celebrity_age_during_season: Standardize (z-score normalization)
        
        Returns:
            DataFrame with standardized numerical features
        """
        if 'celebrity_age_during_season' in self.data.columns:
            scaler = StandardScaler()
            self.data['celebrity_age_standardized'] = scaler.fit_transform(
                self.data[['celebrity_age_during_season']]
            )
            self.scalers['age'] = scaler
            print(f"Standardized celebrity_age_during_season")
        
        return self.data
    
    def create_placement_encoding(self) -> pd.DataFrame:
        """
        Encode placement results:
        - Convert results to numerical placement ranking
        - 1st Place -> 1, 2nd Place -> 2, etc.
        
        Returns:
            DataFrame with encoded placement
        """
        if 'placement' in self.data.columns:
            # Placement already exists as numerical
            pass
        elif 'results' in self.data.columns:
            # Extract placement from results text
            def extract_placement(result_str):
                if pd.isna(result_str):
                    return None
                result_str = str(result_str)
                if '1st Place' in result_str:
                    return 1
                elif '2nd Place' in result_str:
                    return 2
                elif '3rd Place' in result_str:
                    return 3
                elif 'Eliminated Week' in result_str:
                    # Extract week number and assign higher placement
                    week = int(result_str.split('Week')[1].strip())
                    # Earlier elimination = worse placement
                    return week + 3  # Adjust based on season structure
                return None
            
            self.data['placement_derived'] = self.data['results'].apply(extract_placement)
            print("Encoded placement from results")
        
        return self.data
    
    def aggregate_judge_scores(self) -> pd.DataFrame:
        """
        Aggregate judge scores by week
        
        Returns:
            DataFrame with aggregated weekly scores
        """
        # Find all weeks
        weeks = set()
        for col in self.data.columns:
            if 'week' in col and 'judge' in col:
                week_num = col.split('_')[0].replace('week', '')
                weeks.add(int(week_num))
        
        weeks = sorted(list(weeks))
        
        # Create aggregated scores per week
        for week in weeks:
            week_cols = [col for col in self.data.columns 
                        if f'week{week}_judge' in col and 'score' in col]
            if week_cols:
                # Calculate average score for the week
                self.data[f'week{week}_avg_score'] = self.data[week_cols].mean(axis=1)
                # Calculate total score for the week
                self.data[f'week{week}_total_score'] = self.data[week_cols].sum(axis=1)
        
        print(f"Aggregated judge scores for {len(weeks)} weeks")
        return self.data
    
    def preprocess_all(self) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline
        
        Returns:
            Fully preprocessed DataFrame
        """
        print("Starting data preprocessing...")
        
        # Load data
        self.load_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode categorical variables
        self.encode_categorical_variables()
        
        # Standardize numerical features
        self.standardize_numerical_features()
        
        # Create placement encoding
        self.create_placement_encoding()
        
        # Aggregate judge scores
        self.aggregate_judge_scores()
        
        self.processed_data = self.data
        
        print("Preprocessing complete!")
        print(f"Final dataset shape: {self.processed_data.shape}")
        
        return self.processed_data
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Get lists of feature columns by type
        
        Returns:
            Dictionary mapping feature types to column names
        """
        features = {
            'industry': [col for col in self.data.columns if col.startswith('industry_')],
            'judge_scores': [col for col in self.data.columns 
                           if 'week' in col and 'score' in col],
            'categorical_encoded': [col for col in self.data.columns if col.endswith('_encoded')],
            'numerical': ['celebrity_age_during_season', 'celebrity_age_standardized', 
                         'season', 'placement']
        }
        
        return features


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
    processed_data = preprocessor.preprocess_all()
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(processed_data.head())
    
    # Display feature information
    features = preprocessor.get_feature_columns()
    print("\nFeature columns by type:")
    for feature_type, cols in features.items():
        print(f"{feature_type}: {len(cols)} columns")
