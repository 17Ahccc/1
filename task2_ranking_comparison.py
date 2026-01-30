"""
Task 2: Compare ranking method vs percentage method
Evaluates fairness and re-analyzes controversial contestants
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RankingMethodComparator:
    """
    Compares ranking method and percentage method for determining season results
    Assesses fairness and re-analyzes controversial placements
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize ranking method comparator
        
        Args:
            data: Preprocessed DataFrame with contestant information
        """
        self.data = data
        
    def calculate_ranking_method_score(self, contestant_data: pd.Series) -> float:
        """
        Calculate score using ranking method
        In ranking method, judges rank contestants 1, 2, 3, etc.
        Lower rank = better performance
        
        Args:
            contestant_data: Series with contestant's weekly scores
            
        Returns:
            Average ranking score
        """
        # Get all weekly average scores
        score_cols = [col for col in contestant_data.index 
                     if 'week' in col and 'avg_score' in col]
        
        scores = []
        for col in score_cols:
            if contestant_data[col] > 0:  # Only consider non-zero scores
                scores.append(contestant_data[col])
        
        if len(scores) == 0:
            return 0
        
        # Ranking method: convert scores to rankings
        # Assume higher score = better rank (lower number)
        # Simple approximation: rank = 11 - score (for 10-point scale)
        rankings = [11 - s for s in scores]
        avg_ranking = np.mean(rankings)
        
        return avg_ranking
    
    def calculate_percentage_method_score(self, contestant_data: pd.Series) -> float:
        """
        Calculate score using percentage method
        In percentage method, scores are averaged as percentages
        
        Args:
            contestant_data: Series with contestant's weekly scores
            
        Returns:
            Average percentage score
        """
        # Get all weekly average scores
        score_cols = [col for col in contestant_data.index 
                     if 'week' in col and 'avg_score' in col]
        
        scores = []
        for col in score_cols:
            if contestant_data[col] > 0:  # Only consider non-zero scores
                scores.append(contestant_data[col])
        
        if len(scores) == 0:
            return 0
        
        # Percentage method: average scores as percentage of maximum (10)
        max_score = 10.0
        percentage_scores = [(s / max_score) * 100 for s in scores]
        avg_percentage = np.mean(percentage_scores)
        
        return avg_percentage
    
    def compare_methods_by_season(self) -> pd.DataFrame:
        """
        Compare ranking and percentage methods for each season
        
        Returns:
            DataFrame comparing methods across seasons
        """
        results = []
        
        for season in sorted(self.data['season'].unique()):
            season_data = self.data[self.data['season'] == season].copy()
            
            # Calculate scores for each contestant using both methods
            season_data['ranking_method_score'] = season_data.apply(
                self.calculate_ranking_method_score, axis=1
            )
            season_data['percentage_method_score'] = season_data.apply(
                self.calculate_percentage_method_score, axis=1
            )
            
            # Rank contestants by each method (lower rank number = better)
            # For ranking method: lower score is better
            season_data['ranking_method_rank'] = season_data['ranking_method_score'].rank(method='min')
            # For percentage method: higher score is better
            season_data['percentage_method_rank'] = season_data['percentage_method_score'].rank(
                method='min', ascending=False
            )
            
            # Calculate correlation between methods
            correlation = season_data[['ranking_method_rank', 'percentage_method_rank']].corr().iloc[0, 1]
            
            # Calculate agreement (how many contestants have same rank)
            rank_difference = abs(season_data['ranking_method_rank'] - 
                                season_data['percentage_method_rank'])
            avg_rank_diff = rank_difference.mean()
            max_rank_diff = rank_difference.max()
            
            # Identify disagreements (rank difference > 1)
            major_disagreements = len(season_data[rank_difference > 1])
            
            results.append({
                'season': season,
                'num_contestants': len(season_data),
                'correlation': correlation,
                'avg_rank_difference': avg_rank_diff,
                'max_rank_difference': max_rank_diff,
                'major_disagreements': major_disagreements,
                'agreement_rate': (len(season_data) - major_disagreements) / len(season_data)
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n=== Ranking vs Percentage Method Comparison by Season ===")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def assess_fairness(self) -> Dict:
        """
        Assess fairness of both methods using statistical measures
        
        Returns:
            Dictionary with fairness metrics
        """
        # Calculate scores for all contestants
        self.data['ranking_method_score'] = self.data.apply(
            self.calculate_ranking_method_score, axis=1
        )
        self.data['percentage_method_score'] = self.data.apply(
            self.calculate_percentage_method_score, axis=1
        )
        
        # Fairness metric 1: Correlation with actual placement
        ranking_corr = self.data[['ranking_method_score', 'placement']].corr().iloc[0, 1]
        percentage_corr = self.data[['percentage_method_score', 'placement']].corr().iloc[0, 1]
        
        # Fairness metric 2: Variance in scores (lower = more consistent)
        ranking_variance = self.data['ranking_method_score'].var()
        percentage_variance = self.data['percentage_method_score'].var()
        
        # Fairness metric 3: Sensitivity to outliers (using IQR)
        ranking_iqr = self.data['ranking_method_score'].quantile(0.75) - \
                     self.data['ranking_method_score'].quantile(0.25)
        percentage_iqr = self.data['percentage_method_score'].quantile(0.75) - \
                        self.data['percentage_method_score'].quantile(0.25)
        
        # Fairness metric 4: Discrimination power (ability to separate contestants)
        # Coefficient of variation (CV = std/mean)
        ranking_cv = self.data['ranking_method_score'].std() / \
                    self.data['ranking_method_score'].mean()
        percentage_cv = self.data['percentage_method_score'].std() / \
                       self.data['percentage_method_score'].mean()
        
        fairness_metrics = {
            'ranking_method': {
                'correlation_with_placement': ranking_corr,
                'variance': ranking_variance,
                'iqr': ranking_iqr,
                'coefficient_of_variation': ranking_cv
            },
            'percentage_method': {
                'correlation_with_placement': percentage_corr,
                'variance': percentage_variance,
                'iqr': percentage_iqr,
                'coefficient_of_variation': percentage_cv
            }
        }
        
        print("\n=== Fairness Assessment ===")
        print("\nRanking Method:")
        for metric, value in fairness_metrics['ranking_method'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nPercentage Method:")
        for metric, value in fairness_metrics['percentage_method'].items():
            print(f"  {metric}: {value:.4f}")
        
        # Determine which method is fairer
        # Higher correlation with actual placement = fairer
        # Lower variance = more consistent
        # Moderate CV = good discrimination
        
        fairness_score_ranking = (
            abs(ranking_corr) * 0.5 +  # Weight correlation heavily
            (1 / (1 + ranking_variance)) * 0.3 +  # Inverse variance
            ranking_cv * 0.2  # Discrimination power
        )
        
        fairness_score_percentage = (
            abs(percentage_corr) * 0.5 +
            (1 / (1 + percentage_variance)) * 0.3 +
            percentage_cv * 0.2
        )
        
        fairness_metrics['overall_fairness'] = {
            'ranking_method_score': fairness_score_ranking,
            'percentage_method_score': fairness_score_percentage,
            'fairer_method': 'Ranking' if fairness_score_ranking > fairness_score_percentage else 'Percentage'
        }
        
        print(f"\nOverall Fairness Scores:")
        print(f"  Ranking Method: {fairness_score_ranking:.4f}")
        print(f"  Percentage Method: {fairness_score_percentage:.4f}")
        print(f"  Fairer Method: {fairness_metrics['overall_fairness']['fairer_method']}")
        
        return fairness_metrics
    
    def identify_controversial_contestants(self, rank_diff_threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify contestants with controversial rankings
        (large difference between ranking and percentage methods)
        
        Args:
            rank_diff_threshold: Minimum rank difference to be considered controversial
            
        Returns:
            DataFrame with controversial contestants
        """
        # Ensure scores are calculated
        if 'ranking_method_score' not in self.data.columns:
            self.data['ranking_method_score'] = self.data.apply(
                self.calculate_ranking_method_score, axis=1
            )
        if 'percentage_method_score' not in self.data.columns:
            self.data['percentage_method_score'] = self.data.apply(
                self.calculate_percentage_method_score, axis=1
            )
        
        # Calculate ranks for each contestant within their season
        controversial = []
        
        for season in sorted(self.data['season'].unique()):
            season_data = self.data[self.data['season'] == season].copy()
            
            # Calculate ranks
            season_data['ranking_rank'] = season_data['ranking_method_score'].rank(method='min')
            season_data['percentage_rank'] = season_data['percentage_method_score'].rank(
                method='min', ascending=False
            )
            season_data['rank_difference'] = abs(
                season_data['ranking_rank'] - season_data['percentage_rank']
            )
            
            # Identify controversial contestants
            controversial_season = season_data[
                season_data['rank_difference'] >= rank_diff_threshold
            ].copy()
            
            controversial.append(controversial_season)
        
        if len(controversial) > 0:
            controversial_df = pd.concat(controversial, ignore_index=True)
            
            # Select relevant columns
            result_cols = ['celebrity_name', 'season', 'placement', 
                          'ranking_method_score', 'percentage_method_score',
                          'ranking_rank', 'percentage_rank', 'rank_difference']
            controversial_df = controversial_df[result_cols].sort_values(
                'rank_difference', ascending=False
            )
            
            print(f"\n=== Controversial Contestants (Rank Difference >= {rank_diff_threshold}) ===")
            print(controversial_df.to_string(index=False))
            
            return controversial_df
        else:
            print(f"\nNo controversial contestants found with rank difference >= {rank_diff_threshold}")
            return pd.DataFrame()
    
    def reanalyze_contestant(self, contestant_name: str, season: int) -> Dict:
        """
        Detailed re-analysis of a specific contestant
        
        Args:
            contestant_name: Name of contestant to analyze
            season: Season number
            
        Returns:
            Dictionary with detailed analysis
        """
        # Get contestant data
        contestant = self.data[
            (self.data['celebrity_name'] == contestant_name) & 
            (self.data['season'] == season)
        ]
        
        if len(contestant) == 0:
            print(f"Contestant '{contestant_name}' not found in season {season}")
            return {}
        
        contestant = contestant.iloc[0]
        
        # Get weekly performance
        score_cols = [col for col in self.data.columns 
                     if 'week' in col and 'avg_score' in col]
        
        weekly_scores = []
        for col in score_cols:
            week = col.split('_')[0]
            score = contestant[col]
            if score > 0:
                weekly_scores.append({
                    'week': week,
                    'score': score
                })
        
        analysis = {
            'contestant_name': contestant_name,
            'season': season,
            'actual_placement': contestant['placement'],
            'ranking_method_score': contestant.get('ranking_method_score', 0),
            'percentage_method_score': contestant.get('percentage_method_score', 0),
            'weekly_performance': weekly_scores,
            'total_weeks': len(weekly_scores),
            'avg_weekly_score': np.mean([w['score'] for w in weekly_scores]) if weekly_scores else 0,
            'score_trend': 'improving' if len(weekly_scores) > 1 and 
                          weekly_scores[-1]['score'] > weekly_scores[0]['score'] else 'declining'
        }
        
        print(f"\n=== Detailed Analysis: {contestant_name} (Season {season}) ===")
        print(f"Actual Placement: {analysis['actual_placement']}")
        print(f"Ranking Method Score: {analysis['ranking_method_score']:.2f}")
        print(f"Percentage Method Score: {analysis['percentage_method_score']:.2f}")
        print(f"Average Weekly Score: {analysis['avg_weekly_score']:.2f}")
        print(f"Performance Trend: {analysis['score_trend']}")
        print(f"Weeks Competed: {analysis['total_weeks']}")
        
        return analysis


def run_task2_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete Task 2 analysis
    
    Args:
        data: Preprocessed DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("TASK 2: COMPARE RANKING VS PERCENTAGE METHODS")
    print("="*60)
    
    # Create comparator
    comparator = RankingMethodComparator(data)
    
    # Compare methods by season
    season_comparison = comparator.compare_methods_by_season()
    
    # Assess fairness
    fairness_metrics = comparator.assess_fairness()
    
    # Identify controversial contestants
    controversial = comparator.identify_controversial_contestants(rank_diff_threshold=2.0)
    
    # Re-analyze top controversial contestants
    if len(controversial) > 0:
        print("\n=== Detailed Re-analysis of Most Controversial Contestants ===")
        for idx in range(min(3, len(controversial))):
            contestant = controversial.iloc[idx]
            comparator.reanalyze_contestant(
                contestant['celebrity_name'], 
                contestant['season']
            )
    
    return {
        'comparator': comparator,
        'season_comparison': season_comparison,
        'fairness_metrics': fairness_metrics,
        'controversial_contestants': controversial
    }


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
    data = preprocessor.preprocess_all()
    
    # Run Task 2 analysis
    task2_results = run_task2_analysis(data)
    
    # Save results
    if len(task2_results['controversial_contestants']) > 0:
        task2_results['controversial_contestants'].to_csv(
            'task2_controversial_contestants.csv', index=False
        )
        print("\nControversial contestants saved to: task2_controversial_contestants.csv")
    
    task2_results['season_comparison'].to_csv(
        'task2_method_comparison.csv', index=False
    )
    print("Method comparison saved to: task2_method_comparison.csv")
