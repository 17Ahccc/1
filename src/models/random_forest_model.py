"""
Problem 1: Feature Identification using Random Forest and SHAP

This module:
- Trains a Random Forest model
- Calculates SHAP values for feature importance
- Selects top-k features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib
import os


class RandomForestFeatureSelector:
    """Random Forest model with SHAP-based feature selection."""
    
    def __init__(self, task='regression', n_estimators=500, max_depth=10, random_state=42):
        """
        Initialize the Random Forest model.
        
        Args:
            task (str): 'regression' or 'classification'
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            random_state (int): Random seed
        """
        self.task = task
        self.random_state = random_state
        
        if task == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.shap_values = None
        self.feature_importance = None
        self.top_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, df, target_col, test_size=0.2):
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Test set ratio
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Separate features and target
        exclude_cols = [
            'celebrity_name', 'ballroom_partner', 'results', 
            'placement', 'is_eliminated', target_col
        ]
        
        # Remove columns that shouldn't be features
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and not col.startswith('week') or col.endswith('_mean') or col.endswith('_std')]
        
        # Get actual feature columns (exclude raw weekly scores)
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and not (col.startswith('week') and 'judge' in col and 'score' in col)]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        print(f"Features: {list(self.X_train.columns[:10])}... (total: {len(self.X_train.columns)})")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """Train the Random Forest model."""
        print(f"\n=== Training Random Forest ({self.task}) ===")
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        if self.task == 'regression':
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            print(f"Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        else:
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            # AUC score
            if hasattr(self.model, 'predict_proba'):
                train_proba = self.model.predict_proba(self.X_train)[:, 1]
                test_proba = self.model.predict_proba(self.X_test)[:, 1]
                train_auc = roc_auc_score(self.y_train, train_proba)
                test_auc = roc_auc_score(self.y_test, test_proba)
                
                print(f"Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
            else:
                print(f"Train Accuracy: {train_acc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}")
        
        print("=== Training Complete ===\n")
    
    def calculate_shap_values(self, sample_size=100):
        """
        Calculate SHAP values for feature importance.
        
        Args:
            sample_size (int): Number of samples to use for SHAP calculation
        """
        print("\n=== Calculating SHAP Values ===")
        
        # Use a sample for efficiency
        X_sample = self.X_test.sample(min(sample_size, len(self.X_test)), random_state=self.random_state)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X_sample)
        
        # For classification, use positive class SHAP values
        if self.task == 'classification' and isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        print(f"SHAP values calculated for {len(X_sample)} samples")
        print("=== SHAP Calculation Complete ===\n")
        
        return self.shap_values, X_sample
    
    def rank_features(self, top_k=15):
        """
        Rank features by mean absolute SHAP values.
        
        Args:
            top_k (int): Number of top features to select
            
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if self.shap_values is None:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return None
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Select top-k features
        self.top_features = self.feature_importance.head(top_k)['feature'].tolist()
        
        print(f"\n=== Top {top_k} Features ===")
        print(self.feature_importance.head(top_k))
        print()
        
        return self.feature_importance
    
    def plot_feature_importance(self, output_dir='visualizations', top_k=15):
        """
        Plot feature importance using SHAP.
        
        Args:
            output_dir (str): Directory to save plots
            top_k (int): Number of top features to display
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.feature_importance is None:
            print("Feature importance not calculated. Run rank_features() first.")
            return
        
        # Bar plot of feature importance
        plt.figure(figsize=(10, 8))
        top_features_df = self.feature_importance.head(top_k)
        plt.barh(range(len(top_features_df)), top_features_df['importance'])
        plt.yticks(range(len(top_features_df)), top_features_df['feature'])
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title(f'Top {top_k} Feature Importance (Random Forest + SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'feature_importance_shap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {output_path}")
        plt.close()
    
    def save_model(self, model_path='models/random_forest_model.pkl'):
        """Save the trained model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'top_features': self.top_features,
            'feature_columns': self.X_train.columns.tolist()
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/random_forest_model.pkl'):
        """Load a trained model."""
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_importance = data['feature_importance']
        self.top_features = data['top_features']
        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    print("Random Forest Feature Selection Module")
    print("This module should be run as part of the main pipeline.")
