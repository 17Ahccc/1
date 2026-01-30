"""
Feature Engineering Module

This module creates derived features from the raw data:
- judge_score_mean: Average score per week
- judge_score_std: Standard deviation of scores per week
- hist_score_mean: Cumulative average score
- hist_score_std: Cumulative standard deviation
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Creates engineered features for modeling."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def extract_weekly_scores(self, df):
        """
        Extract scores for each week into structured format.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with weekly score features
        """
        df = df.copy()
        
        # Identify all week columns
        max_week = 11  # Based on the data structure
        
        for week in range(1, max_week + 1):
            # Get all judge scores for this week
            week_cols = [col for col in df.columns if f'week{week}_judge' in col and 'score' in col]
            
            if week_cols:
                # Calculate mean score for the week
                df[f'week{week}_mean'] = df[week_cols].mean(axis=1)
                
                # Calculate std score for the week
                df[f'week{week}_std'] = df[week_cols].std(axis=1)
                
                # Replace NaN std with 0 (when only one score exists)
                df[f'week{week}_std'] = df[f'week{week}_std'].fillna(0)
        
        return df
    
    def create_weekly_features(self, df):
        """
        Create judge_score_mean and judge_score_std for each contestant-week.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with weekly features
        """
        df = df.copy()
        
        # First extract weekly scores
        df = self.extract_weekly_scores(df)
        
        # For each contestant, create features for each week they participated
        weekly_data = []
        
        for idx, row in df.iterrows():
            for week in range(1, 12):  # weeks 1-11
                week_mean_col = f'week{week}_mean'
                week_std_col = f'week{week}_std'
                
                if week_mean_col in df.columns and not pd.isna(row[week_mean_col]) and row[week_mean_col] != 0:
                    weekly_record = {
                        'celebrity_name': row['celebrity_name'],
                        'season': row['season'],
                        'placement': row['placement'],
                        'is_eliminated': row['is_eliminated'],
                        'celebrity_age_during_season': row['celebrity_age_during_season'],
                        'week': week,
                        'judge_score_mean': row[week_mean_col],
                        'judge_score_std': row[week_std_col]
                    }
                    
                    # Add categorical features
                    for col in df.columns:
                        if any(cat in col for cat in ['celebrity_industry_', 'ballroom_partner_', 
                                                       'celebrity_homestate_', 'celebrity_homecountry']):
                            weekly_record[col] = row[col]
                    
                    weekly_data.append(weekly_record)
        
        weekly_df = pd.DataFrame(weekly_data)
        print(f"Created {len(weekly_df)} weekly records from {len(df)} contestants")
        
        return weekly_df
    
    def create_historical_features(self, df):
        """
        Create cumulative historical features (hist_score_mean, hist_score_std).
        
        Args:
            df (pd.DataFrame): Weekly dataframe
            
        Returns:
            pd.DataFrame: Dataframe with historical features
        """
        df = df.copy()
        
        # Sort by contestant and week
        df = df.sort_values(['celebrity_name', 'season', 'week'])
        
        # Calculate cumulative mean and std for each contestant
        df['hist_score_mean'] = df.groupby(['celebrity_name', 'season'])['judge_score_mean'].expanding().mean().reset_index(level=[0, 1], drop=True)
        df['hist_score_std'] = df.groupby(['celebrity_name', 'season'])['judge_score_mean'].expanding().std().reset_index(level=[0, 1], drop=True)
        
        # Fill NaN std with 0 (first week has no std)
        df['hist_score_std'] = df['hist_score_std'].fillna(0)
        
        print(f"Created historical features for {len(df)} records")
        
        return df
    
    def create_features(self, df):
        """
        Execute the complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("\n=== Starting Feature Engineering ===")
        
        # Create weekly features
        weekly_df = self.create_weekly_features(df)
        
        # Create historical features
        weekly_df = self.create_historical_features(weekly_df)
        
        print("=== Feature Engineering Complete ===\n")
        
        return weekly_df


if __name__ == "__main__":
    # Example usage
    from data_preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor("../../2026_MCM_Problem_C_Data.csv")
    processed_df = preprocessor.preprocess()
    
    engineer = FeatureEngineer()
    featured_df = engineer.create_features(processed_df)
    
    print(f"\nFinal shape: {featured_df.shape}")
    print(f"\nSample features:\n{featured_df[['celebrity_name', 'week', 'judge_score_mean', 'judge_score_std', 'hist_score_mean', 'hist_score_std']].head(10)}")
