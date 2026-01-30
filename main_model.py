"""
Main Modeling Script for MCM 2026 Problem C
Integrates all tasks and generates comprehensive analysis
"""

import sys
import os
from datetime import datetime

# Import all modules
from data_preprocessing import DataPreprocessor
from task1_voting_estimation import run_task1_analysis
from task2_ranking_comparison import run_task2_analysis
from task3_success_factors import run_task3_analysis
from task4_improved_voting import run_task4_analysis


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def print_section(text):
    """Print formatted section"""
    print("\n" + "-"*70)
    print(text)
    print("-"*70 + "\n")


def main():
    """
    Execute complete modeling pipeline for MCM 2026 Problem C
    """
    print_header("MCM 2026 PROBLEM C: DANCING WITH THE STARS ANALYSIS")
    print_header("MODEL BUILDING MODULE")
    
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data file path
    data_file = '2026_MCM_Problem_C_Data.csv'
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file '{data_file}' not found!")
        print("Please ensure the data file is in the current directory.")
        sys.exit(1)
    
    try:
        # ================================================================
        # DATA PREPROCESSING
        # ================================================================
        print_header("DATA PREPROCESSING")
        
        print("Loading and preprocessing data...")
        preprocessor = DataPreprocessor(data_file)
        data = preprocessor.preprocess_all()
        
        print(f"\nPreprocessed dataset: {data.shape[0]} rows × {data.shape[1]} columns")
        
        # Display variable definitions
        print_section("Variable Definitions")
        print("""
        Core Variables:
        - celebrity_name: Celebrity contestant name
        - ballroom_partner: Professional dancer partner
        - celebrity_industry: Celebrity's profession/industry
        - celebrity_age_during_season: Age during competition
        - season: Season number
        - results: Competition outcome
        - placement: Final ranking (1=winner, higher=eliminated earlier)
        - weekX_judgeY_score: Judge scores by week
        
        Derived Variables:
        - celebrity_age_standardized: Normalized age (z-score)
        - industry_*: One-hot encoded industry categories
        - weekX_avg_score: Average judge score per week
        - weekX_total_score: Total judge score per week
        """)
        
        # Display key assumptions
        print_section("Model Assumptions")
        print("""
        Assumption 1: Missing judge scores filled with 0 (≤5% missing data)
        Assumption 2: Audience voting follows consistent non-skewed distribution
        Assumption 3: Success factors are statistically independent
        Assumption 4: Ranking and percentage methods have no systematic bias
        Assumption 5: AI models (Gradient Boosting) achieve near-optimal accuracy
        """)
        
        # ================================================================
        # TASK 1: ESTIMATE AUDIENCE VOTING NUMBERS
        # ================================================================
        print_header("TASK 1: ESTIMATE AUDIENCE VOTING NUMBERS")
        
        task1_results = run_task1_analysis(data)
        
        # Save Task 1 results
        task1_results['results'].to_csv('task1_voting_estimates.csv', index=False)
        
        print("\n✓ Task 1 completed successfully")
        print(f"  - Model R² Score: {task1_results['metrics']['test_r2']:.4f}")
        print(f"  - RMSE: {task1_results['metrics']['test_rmse']:.4f}M votes")
        print(f"  - Results saved to: task1_voting_estimates.csv")
        
        # ================================================================
        # TASK 2: COMPARE RANKING VS PERCENTAGE METHODS
        # ================================================================
        print_header("TASK 2: COMPARE RANKING VS PERCENTAGE METHODS")
        
        task2_results = run_task2_analysis(data)
        
        # Save Task 2 results
        task2_results['season_comparison'].to_csv('task2_method_comparison.csv', index=False)
        if len(task2_results['controversial_contestants']) > 0:
            task2_results['controversial_contestants'].to_csv('task2_controversial_contestants.csv', index=False)
        
        print("\n✓ Task 2 completed successfully")
        print(f"  - Fairer method: {task2_results['fairness_metrics']['overall_fairness']['fairer_method']}")
        print(f"  - Controversial contestants identified: {len(task2_results['controversial_contestants'])}")
        print(f"  - Results saved to: task2_method_comparison.csv, task2_controversial_contestants.csv")
        
        # ================================================================
        # TASK 3: ANALYZE SUCCESS FACTORS
        # ================================================================
        print_header("TASK 3: ANALYZE FACTORS INFLUENCING SUCCESS")
        
        task3_results = run_task3_analysis(data)
        
        # Save Task 3 results
        task3_results['factor_impacts'].to_csv('task3_factor_impacts.csv', index=False)
        if len(task3_results['industry_analysis']) > 0:
            task3_results['industry_analysis'].to_csv('task3_industry_analysis.csv', index=False)
        if len(task3_results['age_analysis']) > 0:
            task3_results['age_analysis'].to_csv('task3_age_analysis.csv', index=False)
        
        print("\n✓ Task 3 completed successfully")
        print(f"  - Model Accuracy: {task3_results['model_results']['test_accuracy']:.4f}")
        
        # Display top factors
        print("\n  Top 3 Success Factors:")
        top_factors = task3_results['factor_impacts'].head(3)
        for idx, row in top_factors.iterrows():
            print(f"    {idx+1}. {row['feature']}: {row['impact_percentage']:.2f}%")
        
        print(f"  - Results saved to: task3_factor_impacts.csv, task3_industry_analysis.csv, task3_age_analysis.csv")
        
        # ================================================================
        # TASK 4: PROPOSE IMPROVED VOTING SYSTEM
        # ================================================================
        print_header("TASK 4: PROPOSE IMPROVED VOTING SYSTEM")
        
        task4_results = run_task4_analysis(data)
        
        # Save Task 4 results
        hybrid_results = task4_results['hybrid_results'][[
            'celebrity_name', 'season', 'placement', 
            'hybrid_score', 'hybrid_rank', 'rank_change'
        ]].sort_values('hybrid_score', ascending=False)
        hybrid_results.to_csv('task4_hybrid_system_results.csv', index=False)
        
        print("\n✓ Task 4 completed successfully")
        print(f"  - System: {task4_results['hybrid_system']['name']}")
        print(f"  - Correlation with current system: {task4_results['comparison']['correlation_with_current']:.4f}")
        print(f"  - Projected viewership increase: {task4_results['viewership_projections']['total_projection']['increase_percentage']:.1f}%")
        print(f"  - Results saved to: task4_hybrid_system_results.csv")
        
        # ================================================================
        # SUMMARY AND CONCLUSIONS
        # ================================================================
        print_header("ANALYSIS SUMMARY")
        
        print("All tasks completed successfully!\n")
        
        print("Key Findings:")
        print(f"  1. Audience voting patterns estimated with {task1_results['metrics']['test_r2']:.1%} accuracy")
        print(f"  2. {task2_results['fairness_metrics']['overall_fairness']['fairer_method']} method deemed fairer")
        print(f"  3. Top success factor: {task3_results['factor_impacts'].iloc[0]['feature']}")
        print(f"  4. Proposed hybrid system projects {task4_results['viewership_projections']['total_projection']['increase_percentage']:.1f}% viewership increase")
        
        print("\nOutput Files Generated:")
        output_files = [
            'task1_voting_estimates.csv',
            'task2_method_comparison.csv',
            'task2_controversial_contestants.csv',
            'task3_factor_impacts.csv',
            'task3_industry_analysis.csv',
            'task3_age_analysis.csv',
            'task4_hybrid_system_results.csv'
        ]
        
        for file in output_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not generated)")
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print_header("END OF ANALYSIS")
        
        return {
            'task1': task1_results,
            'task2': task2_results,
            'task3': task3_results,
            'task4': task4_results
        }
        
    except Exception as e:
        print(f"\n✗ ERROR: Analysis failed!")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run main analysis
    results = main()
    
    print("\nAll modeling tasks completed successfully!")
    print("Review the generated CSV files for detailed results.")
