"""
Task 3: Analyze factors influencing celebrity success
Quantifies impact of profession, age, industry, etc. on performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SuccessFactorAnalyzer:
    """
    Analyzes characteristics that influence celebrity success in the competition
    Quantifies impact of profession, age, industry, and other features
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize success factor analyzer
        
        Args:
            data: Preprocessed DataFrame with contestant information
        """
        self.data = data
        self.model = None
        self.feature_importance = None
        
    def create_success_labels(self, top_k: int = 3) -> pd.Series:
        """
        Create binary success labels based on placement
        
        Args:
            top_k: Consider top K placements as "successful"
            
        Returns:
            Series with binary success labels (1=successful, 0=not successful)
        """
        # Success = finishing in top K positions
        success = (self.data['placement'] <= top_k).astype(int)
        
        print(f"Success criteria: Top {top_k} placement")
        print(f"Successful contestants: {success.sum()} ({success.mean()*100:.1f}%)")
        print(f"Unsuccessful contestants: {len(success) - success.sum()} ({(1-success.mean())*100:.1f}%)")
        
        return success
    
    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for success analysis
        
        Returns:
            Tuple of (feature DataFrame, list of feature names)
        """
        features_dict = {}
        feature_names = []
        
        # Age features
        if 'celebrity_age_during_season' in self.data.columns:
            features_dict['age'] = self.data['celebrity_age_during_season']
            feature_names.append('age')
            
            # Age squared (to capture non-linear effects)
            features_dict['age_squared'] = self.data['celebrity_age_during_season'] ** 2
            feature_names.append('age_squared')
        
        # Season (to account for time effects)
        if 'season' in self.data.columns:
            features_dict['season'] = self.data['season']
            feature_names.append('season')
        
        # Industry features (one-hot encoded)
        industry_cols = [col for col in self.data.columns if col.startswith('industry_')]
        for col in industry_cols:
            features_dict[col] = self.data[col]
            feature_names.append(col)
        
        # Average judge score (as a performance indicator)
        score_cols = [col for col in self.data.columns 
                     if 'week' in col and 'avg_score' in col]
        if score_cols:
            # Calculate overall average score
            score_matrix = self.data[score_cols].replace(0, np.nan)
            features_dict['overall_avg_score'] = score_matrix.mean(axis=1)
            feature_names.append('overall_avg_score')
            
            # Score consistency (standard deviation)
            features_dict['score_consistency'] = score_matrix.std(axis=1)
            feature_names.append('score_consistency')
            
            # Score trend (improvement over time)
            # Calculate correlation between week number and score
            score_trends = []
            for idx, row in score_matrix.iterrows():
                valid_scores = [(i+1, score) for i, score in enumerate(row) if not pd.isna(score)]
                if len(valid_scores) > 1:
                    weeks, scores = zip(*valid_scores)
                    correlation = np.corrcoef(weeks, scores)[0, 1]
                    score_trends.append(correlation if not np.isnan(correlation) else 0)
                else:
                    score_trends.append(0)
            features_dict['score_trend'] = score_trends
            feature_names.append('score_trend')
        
        # Create feature DataFrame
        X = pd.DataFrame(features_dict)
        X = X.fillna(0)
        
        print(f"\nPrepared {len(feature_names)} features for analysis")
        
        return X, feature_names
    
    def analyze_correlation(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Analyze correlation between features and success
        
        Args:
            X: Feature matrix
            y: Success labels
            
        Returns:
            DataFrame with correlation coefficients
        """
        correlations = []
        
        for col in X.columns:
            # Calculate Pearson correlation
            corr, p_value = stats.pearsonr(X[col], y)
            
            # Calculate point-biserial correlation (for binary outcome)
            # This is actually the same as Pearson for binary y
            
            correlations.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value,
                'significant': 'Yes' if p_value < 0.05 else 'No'
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        
        print("\n=== Feature Correlation with Success ===")
        print(corr_df.to_string(index=False))
        
        return corr_df
    
    def train_success_model(self, X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train model to predict success based on features
        
        Args:
            X: Feature matrix
            y: Success labels
            test_size: Proportion for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train Gradient Boosting model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }
        
        print("\n=== Success Prediction Model Results ===")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("\nClassification Report:")
        print(results['classification_report'])
        
        return results
    
    def quantify_factor_impact(self) -> pd.DataFrame:
        """
        Quantify the impact of each factor on success
        
        Returns:
            DataFrame with quantified impacts
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train_success_model() first.")
        
        # Get feature importance
        impact_df = self.feature_importance.copy()
        
        # Normalize importance to percentages
        impact_df['impact_percentage'] = (impact_df['importance'] / 
                                         impact_df['importance'].sum() * 100)
        
        # Categorize features
        def categorize_feature(feature):
            if 'age' in feature.lower():
                return 'Age'
            elif 'industry_' in feature.lower():
                return 'Industry/Profession'
            elif 'season' in feature.lower():
                return 'Season/Time'
            elif 'score' in feature.lower():
                return 'Performance'
            else:
                return 'Other'
        
        impact_df['category'] = impact_df['feature'].apply(categorize_feature)
        
        print("\n=== Quantified Factor Impact on Success ===")
        print(impact_df.to_string(index=False))
        
        # Aggregate by category
        category_impact = impact_df.groupby('category')['impact_percentage'].sum().sort_values(ascending=False)
        print("\n=== Impact by Category ===")
        for category, impact in category_impact.items():
            print(f"{category}: {impact:.2f}%")
        
        return impact_df
    
    def analyze_by_industry(self, y: pd.Series) -> pd.DataFrame:
        """
        Analyze success rates by industry
        
        Args:
            y: Success labels
            
        Returns:
            DataFrame with industry-specific analysis
        """
        if 'celebrity_industry' not in self.data.columns:
            print("Industry information not available")
            return pd.DataFrame()
        
        # Calculate success rate by industry
        industry_analysis = []
        
        for industry in self.data['celebrity_industry'].unique():
            if pd.isna(industry):
                continue
            
            industry_mask = self.data['celebrity_industry'] == industry
            industry_data = self.data[industry_mask]
            industry_success = y[industry_mask]
            
            analysis = {
                'industry': industry,
                'count': len(industry_data),
                'success_count': industry_success.sum(),
                'success_rate': industry_success.mean(),
                'avg_age': industry_data['celebrity_age_during_season'].mean(),
                'avg_placement': industry_data['placement'].mean()
            }
            
            industry_analysis.append(analysis)
        
        industry_df = pd.DataFrame(industry_analysis).sort_values('success_rate', ascending=False)
        
        print("\n=== Success Analysis by Industry ===")
        print(industry_df.to_string(index=False))
        
        return industry_df
    
    def analyze_by_age_group(self, y: pd.Series) -> pd.DataFrame:
        """
        Analyze success rates by age group
        
        Args:
            y: Success labels
            
        Returns:
            DataFrame with age group analysis
        """
        if 'celebrity_age_during_season' not in self.data.columns:
            print("Age information not available")
            return pd.DataFrame()
        
        # Define age groups
        age_bins = [0, 30, 40, 50, 100]
        age_labels = ['<30', '30-40', '40-50', '50+']
        
        self.data['age_group'] = pd.cut(self.data['celebrity_age_during_season'], 
                                        bins=age_bins, labels=age_labels)
        
        # Analyze by age group
        age_analysis = []
        
        for age_group in age_labels:
            age_mask = self.data['age_group'] == age_group
            age_data = self.data[age_mask]
            age_success = y[age_mask]
            
            if len(age_data) > 0:
                analysis = {
                    'age_group': age_group,
                    'count': len(age_data),
                    'success_count': age_success.sum(),
                    'success_rate': age_success.mean(),
                    'avg_placement': age_data['placement'].mean()
                }
                age_analysis.append(analysis)
        
        age_df = pd.DataFrame(age_analysis)
        
        print("\n=== Success Analysis by Age Group ===")
        print(age_df.to_string(index=False))
        
        return age_df


def run_task3_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete Task 3 analysis
    
    Args:
        data: Preprocessed DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("TASK 3: ANALYZE SUCCESS FACTORS")
    print("="*60)
    
    # Create analyzer
    analyzer = SuccessFactorAnalyzer(data)
    
    # Create success labels (top 3 = successful)
    y = analyzer.create_success_labels(top_k=3)
    
    # Prepare features
    X, feature_names = analyzer.prepare_features()
    
    # Analyze correlations
    correlations = analyzer.analyze_correlation(X, y)
    
    # Train success prediction model
    model_results = analyzer.train_success_model(X, y)
    
    # Quantify factor impacts
    factor_impacts = analyzer.quantify_factor_impact()
    
    # Analyze by industry
    industry_analysis = analyzer.analyze_by_industry(y)
    
    # Analyze by age group
    age_analysis = analyzer.analyze_by_age_group(y)
    
    return {
        'analyzer': analyzer,
        'correlations': correlations,
        'model_results': model_results,
        'factor_impacts': factor_impacts,
        'industry_analysis': industry_analysis,
        'age_analysis': age_analysis
    }


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
    data = preprocessor.preprocess_all()
    
    # Run Task 3 analysis
    task3_results = run_task3_analysis(data)
    
    # Save results
    task3_results['factor_impacts'].to_csv('task3_factor_impacts.csv', index=False)
    print("\nFactor impacts saved to: task3_factor_impacts.csv")
    
    if len(task3_results['industry_analysis']) > 0:
        task3_results['industry_analysis'].to_csv('task3_industry_analysis.csv', index=False)
        print("Industry analysis saved to: task3_industry_analysis.csv")
    
    task3_results['age_analysis'].to_csv('task3_age_analysis.csv', index=False)
    print("Age analysis saved to: task3_age_analysis.csv")
