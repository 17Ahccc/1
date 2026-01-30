"""
Main Execution Script for Dancing with the Stars Analysis

This script orchestrates the complete modeling pipeline:
1. Data preprocessing and feature engineering
2. Feature identification (Random Forest + SHAP)
3. Ranking prediction and elimination classification (XGBoost)
4. Fairness analysis (ANOVA + LMM)
5. Trend forecasting (Prophet)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.feature_engineer import FeatureEngineer
from models.random_forest_model import RandomForestFeatureSelector
from models.xgboost_model import XGBoostMultiTask
from analysis.fairness_analyzer import FairnessAnalyzer
from analysis.trend_forecaster import TrendForecaster

import pandas as pd
import numpy as np


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Execute the complete modeling pipeline."""
    
    print_section_header("DANCING WITH THE STARS - MODELING PROCESS")
    print("This script implements a comprehensive analysis pipeline including:")
    print("  1. Feature Identification (Random Forest + SHAP)")
    print("  2. Ranking Prediction & Elimination Classification (XGBoost)")
    print("  3. Fairness Analysis (ANOVA + LMM)")
    print("  4. Trend Forecasting (Prophet)")
    print()
    
    # Configuration
    DATA_PATH = '2026_MCM_Problem_C_Data.csv'
    OUTPUT_DIRS = {
        'models': 'models',
        'visualizations': 'visualizations',
        'reports': 'reports',
        'data': 'data'
    }
    
    # Create output directories
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print_section_header("STEP 1: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor(DATA_PATH)
    processed_df = preprocessor.preprocess()
    
    # Save processed data
    processed_df.to_csv(os.path.join(OUTPUT_DIRS['data'], 'processed_data.csv'), index=False)
    print("Processed data saved to data/processed_data.csv")
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print_section_header("STEP 2: FEATURE ENGINEERING")
    
    engineer = FeatureEngineer()
    featured_df = engineer.create_features(processed_df)
    
    # Save featured data
    featured_df.to_csv(os.path.join(OUTPUT_DIRS['data'], 'featured_data.csv'), index=False)
    print("Featured data saved to data/featured_data.csv")
    
    # =========================================================================
    # STEP 3: PROBLEM 1 - FEATURE IDENTIFICATION (Random Forest + SHAP)
    # =========================================================================
    print_section_header("PROBLEM 1: FEATURE IDENTIFICATION (Random Forest + SHAP)")
    
    # Train Random Forest for placement prediction
    rf_model = RandomForestFeatureSelector(task='regression', n_estimators=500, max_depth=10)
    rf_model.prepare_data(featured_df, target_col='placement')
    rf_model.train()
    
    # Calculate SHAP values
    shap_values, X_sample = rf_model.calculate_shap_values(sample_size=100)
    
    # Rank features and select top-k
    feature_importance = rf_model.rank_features(top_k=15)
    top_features = rf_model.top_features
    
    # Plot and save
    rf_model.plot_feature_importance(output_dir=OUTPUT_DIRS['visualizations'], top_k=15)
    rf_model.save_model(os.path.join(OUTPUT_DIRS['models'], 'random_forest_model.pkl'))
    
    print(f"\nSelected top {len(top_features)} features:")
    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature}")
    
    # =========================================================================
    # STEP 4: PROBLEM 2 - RANKING PREDICTION & ELIMINATION (XGBoost)
    # =========================================================================
    print_section_header("PROBLEM 2: RANKING PREDICTION & ELIMINATION (XGBoost)")
    
    # Train XGBoost models using top features
    xgb_model = XGBoostMultiTask()
    xgb_model.prepare_data(featured_df, feature_cols=top_features)
    
    # Train regression model
    reg_metrics = xgb_model.train_regression()
    
    # Train classification model
    clf_metrics = xgb_model.train_classification()
    
    # Calculate residuals
    df_with_residuals = xgb_model.calculate_residuals(featured_df, top_features)
    
    # Save data with residuals
    df_with_residuals.to_csv(os.path.join(OUTPUT_DIRS['data'], 'data_with_residuals.csv'), index=False)
    
    # Plot predictions
    xgb_model.plot_predictions(output_dir=OUTPUT_DIRS['visualizations'])
    xgb_model.save_models(output_dir=OUTPUT_DIRS['models'])
    
    print("\nModel Performance:")
    print(f"  Regression - Test RMSE: {reg_metrics['test_rmse']:.4f}, R²: {reg_metrics['test_r2']:.4f}")
    print(f"  Classification - Test Accuracy: {clf_metrics['test_acc']:.4f}, AUC: {clf_metrics['test_auc']:.4f}")
    
    # =========================================================================
    # STEP 5: PROBLEM 3 - FAIRNESS ANALYSIS (ANOVA + LMM)
    # =========================================================================
    print_section_header("PROBLEM 3: FAIRNESS ANALYSIS (ANOVA + LMM)")
    
    fairness_analyzer = FairnessAnalyzer()
    
    # Prepare data for fairness analysis
    fairness_df = fairness_analyzer.prepare_fairness_data(df_with_residuals)
    
    # Perform ANOVA tests
    industry_anova = fairness_analyzer.perform_anova_industry(fairness_df)
    season_anova = fairness_analyzer.perform_anova_season(fairness_df)
    
    # Fit Linear Mixed Model
    lmm_results = fairness_analyzer.fit_linear_mixed_model(fairness_df)
    
    # Generate visualizations and report
    fairness_analyzer.plot_fairness_analysis(fairness_df, output_dir=OUTPUT_DIRS['visualizations'])
    fairness_analyzer.generate_fairness_report(output_dir=OUTPUT_DIRS['reports'])
    
    # =========================================================================
    # STEP 6: PROBLEM 4 - TREND FORECASTING (Prophet)
    # =========================================================================
    print_section_header("PROBLEM 4: TREND FORECASTING (Prophet)")
    
    forecaster = TrendForecaster()
    
    # Aggregate data by season
    season_metrics = forecaster.aggregate_by_season(df_with_residuals)
    
    # Forecast key metrics
    metrics_to_forecast = [
        ('avg_score', 'Average Score'),
        ('fairness_index', 'Fairness Index'),
        ('elimination_rate', 'Elimination Rate')
    ]
    
    for metric_col, metric_name in metrics_to_forecast:
        print(f"\nForecasting: {metric_name}")
        
        # Prepare time series
        ts_data = forecaster.prepare_time_series(season_metrics, metric_col)
        
        # Train Prophet model
        model = forecaster.train_prophet_model(ts_data, metric_name, changepoint_years=[2032])
        
        # Generate forecast
        forecast = forecaster.forecast_future(metric_name, periods=5)
        
        # Plot forecast
        forecaster.plot_forecast(metric_name, output_dir=OUTPUT_DIRS['visualizations'])
        
        # Analyze trends
        forecaster.analyze_trend_changes(metric_name)
    
    # Generate forecast report
    forecaster.generate_forecast_report(output_dir=OUTPUT_DIRS['reports'])
    
    # =========================================================================
    # STEP 7: SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    print_section_header("SUMMARY AND RECOMMENDATIONS")
    
    print("FINDINGS:")
    print("-" * 80)
    
    print("\n1. FEATURE IMPORTANCE:")
    print(f"   - Top 3 features: {', '.join(top_features[:3])}")
    print("   - These features have the most impact on contestant placement")
    
    print("\n2. MODEL PERFORMANCE:")
    print(f"   - Placement prediction: RMSE = {reg_metrics['test_rmse']:.4f}")
    print(f"   - Elimination prediction: AUC = {clf_metrics['test_auc']:.4f}")
    
    print("\n3. FAIRNESS ANALYSIS:")
    if industry_anova.get('significant', False):
        print("   - BIAS DETECTED: Industry has a significant effect on residuals")
    else:
        print("   - No significant industry bias detected")
    
    if season_anova.get('significant', False):
        print("   - BIAS DETECTED: Season has a significant effect on residuals")
    else:
        print("   - No significant season bias detected")
    
    print("\n4. TREND FORECASTS:")
    print("   - Forecasts generated for next 3-5 seasons")
    print("   - See visualizations/ folder for detailed plots")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    print("""
1. JUDGING SYSTEM:
   - Consider weighting features identified as most important
   - Implement standardized scoring rubrics based on top features
   - Regular training for judges to reduce subjectivity

2. FAIRNESS IMPROVEMENTS:
   - Monitor and address any detected biases (industry/season)
   - Ensure diverse representation across all categories
   - Implement blind scoring for certain components

3. VOTING SYSTEM PROPOSAL:
   - Combine judge scores (60%) with audience votes (40%)
   - Use top features to create a weighted scoring system
   - Implement fairness adjustments based on residual analysis

4. FUTURE MONITORING:
   - Track metrics against forecasted trends
   - Reassess fairness quarterly
   - Update models with new season data
""")
    
    print("\n" + "="*80)
    print("DELIVERABLES GENERATED:")
    print("="*80)
    print("""
✓ Code Implementation: Complete pipeline in src/
✓ Visualizations:
  - visualizations/feature_importance_shap.png
  - visualizations/xgboost_predictions.png
  - visualizations/fairness_analysis.png
  - visualizations/forecast_*.png
✓ Reports:
  - reports/fairness_analysis_report.txt
  - reports/trend_forecast_report.txt
✓ Data:
  - data/processed_data.csv
  - data/featured_data.csv
  - data/data_with_residuals.csv
✓ Models:
  - models/random_forest_model.pkl
  - models/xgboost_regressor.pkl
  - models/xgboost_classifier.pkl
""")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
