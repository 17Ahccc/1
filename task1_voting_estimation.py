"""
Task 1: Estimate audience voting numbers
Determines prediction confidence intervals for audience votes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class VotingEstimator:
    """
    Estimates audience voting numbers based on contestant features and judge scores
    Provides confidence intervals for predictions
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize voting estimator
        
        Args:
            data: Preprocessed DataFrame with contestant information
        """
        self.data = data
        self.model = None
        self.feature_importance = None
        
    def create_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create feature matrix for voting estimation
        Uses judge scores, placement, age, season, and industry
        
        Returns:
            Tuple of (X features, y target - estimated as inverse of placement)
        """
        # Select relevant features
        feature_cols = []
        
        # Add judge score features
        score_cols = [col for col in self.data.columns 
                     if 'week' in col and ('avg_score' in col or 'total_score' in col)]
        feature_cols.extend(score_cols)
        
        # Add season
        if 'season' in self.data.columns:
            feature_cols.append('season')
        
        # Add age (standardized)
        if 'celebrity_age_standardized' in self.data.columns:
            feature_cols.append('celebrity_age_standardized')
        
        # Add industry features
        industry_cols = [col for col in self.data.columns if col.startswith('industry_')]
        feature_cols.extend(industry_cols)
        
        # Create feature matrix
        X = self.data[feature_cols].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Create target variable: estimated fan votes (inverse relationship with placement)
        # Better placement (lower number) = more votes
        # Using exponential decay: votes ∝ e^(-k*placement)
        if 'placement' in self.data.columns:
            # Normalize placement to create vote estimates
            # Assume 1st place gets 10M votes, exponentially decreasing
            base_votes = 10.0  # millions
            decay_rate = 0.3
            y = base_votes * np.exp(-decay_rate * self.data['placement'])
        else:
            # If no placement, use inverse of derived placement
            y = base_votes * np.exp(-decay_rate * self.data['placement_derived'])
        
        print(f"Created feature matrix with {X.shape[1]} features")
        print(f"Target variable (estimated fan votes) range: {y.min():.2f}M - {y.max():.2f}M")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train voting estimation model using ensemble methods
        
        Args:
            X: Feature matrix
            y: Target variable (vote estimates)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train Gradient Boosting model (similar to XGBoost as per Assumption 5)
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            subsample=0.8
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, 
                                   scoring='r2')
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std()
        }
        
        print("\n=== Model Training Results ===")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}M votes")
        print(f"Test RMSE: {test_rmse:.4f}M votes")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return metrics
    
    def predict_with_confidence(self, X: pd.DataFrame, 
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict voting numbers with confidence intervals
        
        Args:
            X: Feature matrix for prediction
            confidence_level: Confidence level for intervals (default 95%)
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Estimate prediction intervals using quantile regression approach
        # Use residuals from training to estimate uncertainty
        X_train = self.data[[col for col in X.columns if col in self.data.columns]]
        X_train = X_train.fillna(0)
        
        # Calculate residuals
        y_true_train = self.create_features()[1]
        y_pred_train = self.model.predict(X_train)
        residuals = y_true_train - y_pred_train
        
        # Estimate standard error
        std_error = np.std(residuals)
        
        # Calculate confidence intervals using t-distribution
        alpha = 1 - confidence_level
        df = len(residuals) - 1
        t_value = stats.t.ppf(1 - alpha/2, df)
        
        margin_of_error = t_value * std_error
        
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        # Ensure non-negative votes
        lower_bound = np.maximum(lower_bound, 0)
        
        print(f"\n=== Prediction Confidence Intervals ({confidence_level*100}%) ===")
        print(f"Average prediction: {predictions.mean():.2f}M votes")
        print(f"Average margin of error: ±{margin_of_error:.2f}M votes")
        print(f"Confidence interval width: {2*margin_of_error:.2f}M votes")
        
        return predictions, lower_bound, upper_bound
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features for vote prediction
        
        Args:
            n: Number of top features to return
            
        Returns:
            DataFrame with top features and their importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        return self.feature_importance.head(n)
    
    def estimate_all_votes(self) -> pd.DataFrame:
        """
        Estimate votes for all contestants in dataset
        
        Returns:
            DataFrame with contestant info and estimated votes with confidence intervals
        """
        X, y = self.create_features()
        
        # Train model if not already trained
        if self.model is None:
            self.train_model(X, y)
        
        # Predict with confidence intervals
        predictions, lower, upper = self.predict_with_confidence(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'celebrity_name': self.data['celebrity_name'],
            'season': self.data['season'],
            'placement': self.data['placement'],
            'estimated_fan_votes_millions': predictions,
            'lower_bound_95ci': lower,
            'upper_bound_95ci': upper,
            'confidence_interval_width': upper - lower
        })
        
        # Sort by estimated votes (descending)
        results = results.sort_values('estimated_fan_votes_millions', ascending=False)
        
        return results


def run_task1_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete Task 1 analysis
    
    Args:
        data: Preprocessed DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("TASK 1: ESTIMATE AUDIENCE VOTING NUMBERS")
    print("="*60)
    
    # Create estimator
    estimator = VotingEstimator(data)
    
    # Create features and train model
    X, y = estimator.create_features()
    metrics = estimator.train_model(X, y)
    
    # Get feature importance
    print("\n=== Top 10 Features for Vote Prediction ===")
    top_features = estimator.get_top_features(10)
    print(top_features.to_string(index=False))
    
    # Estimate votes for all contestants
    results = estimator.estimate_all_votes()
    
    print("\n=== Sample Vote Estimates (Top 10) ===")
    print(results.head(10).to_string(index=False))
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average estimated votes: {results['estimated_fan_votes_millions'].mean():.2f}M")
    print(f"Median estimated votes: {results['estimated_fan_votes_millions'].median():.2f}M")
    print(f"Std dev of estimates: {results['estimated_fan_votes_millions'].std():.2f}M")
    print(f"Average confidence interval width: {results['confidence_interval_width'].mean():.2f}M")
    
    return {
        'estimator': estimator,
        'metrics': metrics,
        'results': results,
        'feature_importance': top_features
    }


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
    data = preprocessor.preprocess_all()
    
    # Run Task 1 analysis
    task1_results = run_task1_analysis(data)
    
    # Save results
    task1_results['results'].to_csv('task1_voting_estimates.csv', index=False)
    print("\nResults saved to: task1_voting_estimates.csv")
