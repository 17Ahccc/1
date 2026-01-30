"""
Problem 2: Ranking Prediction & Elimination Classification using XGBoost

This module:
- Trains XGBoost regressor for placement prediction
- Trains XGBoost classifier for elimination prediction
- Calculates residuals for fairness analysis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import joblib
import os


class XGBoostMultiTask:
    """XGBoost models for ranking and elimination prediction."""
    
    def __init__(self, random_state=42):
        """
        Initialize XGBoost models.
        
        Args:
            random_state (int): Random seed
        """
        self.random_state = random_state
        
        # Regression model for placement
        self.regressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            early_stopping_rounds=10
        )
        
        # Classification model for elimination
        self.classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            early_stopping_rounds=10,
            eval_metric='logloss'
        )
        
        self.X_train = None
        self.X_test = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.y_train_clf = None
        self.y_test_clf = None
        
        self.placement_predictions = None
        self.elimination_predictions = None
        self.residuals = None
    
    def prepare_data(self, df, feature_cols, test_size=0.2):
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_cols (list): List of feature column names
            test_size (float): Test set ratio
            
        Returns:
            tuple: Training and test sets
        """
        # Filter to only use specified features
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < len(feature_cols):
            print(f"Warning: Only {len(available_features)}/{len(feature_cols)} features available")
        
        X = df[available_features]
        y_placement = df['placement']
        y_elimination = df['is_eliminated']
        
        # Remove rows with NaN in targets
        mask = ~(y_placement.isna() | y_elimination.isna())
        X = X[mask]
        y_placement = y_placement[mask]
        y_elimination = y_elimination[mask]
        
        # Split data (same split for both tasks)
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X, y_placement, test_size=test_size, random_state=self.random_state
        )
        
        _, _, y_train_clf, y_test_clf = train_test_split(
            X, y_elimination, test_size=test_size, random_state=self.random_state
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_reg = y_train_reg
        self.y_test_reg = y_test_reg
        self.y_train_clf = y_train_clf
        self.y_test_clf = y_test_clf
        
        print(f"Data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(available_features)}")
        print(f"  Elimination rate: {y_elimination.mean():.2%}")
        
        return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf
    
    def train_regression(self):
        """Train XGBoost regressor for placement prediction."""
        print("\n=== Training XGBoost Regressor (Placement) ===")
        
        # Train with validation set for early stopping
        self.regressor.fit(
            self.X_train, 
            self.y_train_reg,
            eval_set=[(self.X_test, self.y_test_reg)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.regressor.predict(self.X_train)
        test_pred = self.regressor.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train_reg, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test_reg, test_pred))
        train_r2 = r2_score(self.y_train_reg, train_pred)
        test_r2 = r2_score(self.y_test_reg, test_pred)
        
        print(f"Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        print("=== Regression Training Complete ===\n")
        
        return {'train_rmse': train_rmse, 'test_rmse': test_rmse, 
                'train_r2': train_r2, 'test_r2': test_r2}
    
    def train_classification(self):
        """Train XGBoost classifier for elimination prediction."""
        print("\n=== Training XGBoost Classifier (Elimination) ===")
        
        # Train with validation set for early stopping
        self.classifier.fit(
            self.X_train, 
            self.y_train_clf,
            eval_set=[(self.X_test, self.y_test_clf)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.classifier.predict(self.X_train)
        test_pred = self.classifier.predict(self.X_test)
        
        train_pred_proba = self.classifier.predict_proba(self.X_train)[:, 1]
        test_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]
        
        train_acc = accuracy_score(self.y_train_clf, train_pred)
        test_acc = accuracy_score(self.y_test_clf, test_pred)
        
        train_auc = roc_auc_score(self.y_train_clf, train_pred_proba)
        test_auc = roc_auc_score(self.y_test_clf, test_pred_proba)
        
        print(f"Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print("=== Classification Training Complete ===\n")
        
        return {'train_acc': train_acc, 'test_acc': test_acc,
                'train_auc': train_auc, 'test_auc': test_auc}
    
    def calculate_residuals(self, df, feature_cols):
        """
        Calculate residuals for fairness analysis.
        
        Args:
            df (pd.DataFrame): Full dataset
            feature_cols (list): Feature columns
            
        Returns:
            pd.DataFrame: Dataset with residuals
        """
        print("\n=== Calculating Residuals ===")
        
        # Get features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features]
        
        # Predict placements
        predictions = self.regressor.predict(X)
        
        # Calculate residuals
        residuals = df['placement'] - predictions
        
        # Add to dataframe
        df_with_residuals = df.copy()
        df_with_residuals['predicted_placement'] = predictions
        df_with_residuals['residual'] = residuals
        
        print(f"Residuals calculated. Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
        print("=== Residuals Calculation Complete ===\n")
        
        self.residuals = residuals
        return df_with_residuals
    
    def plot_predictions(self, output_dir='visualizations'):
        """
        Plot prediction results.
        
        Args:
            output_dir (str): Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot actual vs predicted placement
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Regression plot
        test_pred = self.regressor.predict(self.X_test)
        axes[0].scatter(self.y_test_reg, test_pred, alpha=0.6)
        axes[0].plot([self.y_test_reg.min(), self.y_test_reg.max()], 
                     [self.y_test_reg.min(), self.y_test_reg.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Placement')
        axes[0].set_ylabel('Predicted Placement')
        axes[0].set_title('Placement Prediction (XGBoost Regression)')
        axes[0].grid(True, alpha=0.3)
        
        # Classification ROC curve placeholder (simplified)
        test_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.y_test_clf, test_pred_proba)
        axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(self.y_test_clf, test_pred_proba):.3f}')
        axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Elimination Prediction (XGBoost Classification)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'xgboost_predictions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction plots to {output_path}")
        plt.close()
    
    def save_models(self, output_dir='models'):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        reg_path = os.path.join(output_dir, 'xgboost_regressor.pkl')
        clf_path = os.path.join(output_dir, 'xgboost_classifier.pkl')
        
        joblib.dump(self.regressor, reg_path)
        joblib.dump(self.classifier, clf_path)
        
        print(f"Models saved to {output_dir}/")


if __name__ == "__main__":
    print("XGBoost Multi-Task Module")
    print("This module should be run as part of the main pipeline.")
