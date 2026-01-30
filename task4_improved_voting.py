"""
Task 4: Propose improved voting system
Designs enhanced voting mechanism for fairness and viewership
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ImprovedVotingSystem:
    """
    Proposes and evaluates improved voting system
    Enhances fairness and viewership experience
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize improved voting system designer
        
        Args:
            data: Preprocessed DataFrame with contestant information
        """
        self.data = data
        
    def current_system_analysis(self) -> Dict:
        """
        Analyze issues with current voting system
        
        Returns:
            Dictionary with current system analysis
        """
        issues = {
            'judge_weight': {
                'description': 'Judge scores heavily influence results',
                'impact': 'May overshadow audience preferences',
                'severity': 'Medium'
            },
            'ranking_inconsistency': {
                'description': 'Ranking vs percentage methods produce different results',
                'impact': 'Reduces predictability and fairness perception',
                'severity': 'High'
            },
            'popularity_bias': {
                'description': 'Celebrity popularity may override performance',
                'impact': 'Less skilled contestants may advance',
                'severity': 'Medium'
            },
            'vote_transparency': {
                'description': 'Actual vote counts not disclosed',
                'impact': 'Reduces trust in results',
                'severity': 'Low'
            }
        }
        
        print("=== Current System Issues ===")
        for issue_name, issue_info in issues.items():
            print(f"\n{issue_name}:")
            print(f"  Description: {issue_info['description']}")
            print(f"  Impact: {issue_info['impact']}")
            print(f"  Severity: {issue_info['severity']}")
        
        return issues
    
    def design_hybrid_system(self) -> Dict:
        """
        Design hybrid voting system combining multiple methods
        
        Returns:
            Dictionary with system specifications
        """
        system = {
            'name': 'Hybrid Weighted Voting System',
            'components': {
                'judge_score': {
                    'weight': 0.40,
                    'description': 'Technical performance evaluation by experts',
                    'calculation': 'Average of judge scores (percentage method)'
                },
                'audience_vote': {
                    'weight': 0.35,
                    'description': 'Popular vote from viewers',
                    'calculation': 'Percentage of total votes received'
                },
                'performance_trend': {
                    'weight': 0.15,
                    'description': 'Improvement trajectory over weeks',
                    'calculation': 'Correlation between week number and scores'
                },
                'consistency_bonus': {
                    'weight': 0.10,
                    'description': 'Reward for consistent high performance',
                    'calculation': 'Inverse of score standard deviation'
                }
            },
            'features': [
                'Transparent vote counts',
                'Real-time score updates',
                'Anti-manipulation safeguards',
                'Demographic vote balancing'
            ],
            'fairness_enhancements': [
                'Reduces single-factor dominance',
                'Rewards consistent improvement',
                'Balances expert and popular opinion',
                'Increases transparency'
            ]
        }
        
        print("\n=== Proposed Hybrid Voting System ===")
        print(f"Name: {system['name']}")
        print("\nComponents:")
        for component, details in system['components'].items():
            print(f"\n  {component} ({details['weight']*100:.0f}% weight):")
            print(f"    - {details['description']}")
            print(f"    - Calculation: {details['calculation']}")
        
        print("\nKey Features:")
        for feature in system['features']:
            print(f"  - {feature}")
        
        print("\nFairness Enhancements:")
        for enhancement in system['fairness_enhancements']:
            print(f"  - {enhancement}")
        
        return system
    
    def calculate_hybrid_scores(self, hybrid_system: Dict) -> pd.DataFrame:
        """
        Calculate scores using proposed hybrid system
        
        Args:
            hybrid_system: System specification dictionary
            
        Returns:
            DataFrame with hybrid scores
        """
        results = self.data.copy()
        
        # Component 1: Judge Score (percentage method)
        score_cols = [col for col in self.data.columns 
                     if 'week' in col and 'avg_score' in col]
        if score_cols:
            score_matrix = self.data[score_cols].replace(0, np.nan)
            # Normalize to 0-100 scale (assuming 10-point max)
            results['judge_component'] = (score_matrix.mean(axis=1) / 10.0) * 100
            results['judge_component'] = results['judge_component'].fillna(0)
        else:
            results['judge_component'] = 0
        
        # Component 2: Audience Vote (estimated from placement)
        # Better placement = more votes (exponential)
        base_votes = 100.0
        decay_rate = 0.25
        results['audience_component'] = base_votes * np.exp(-decay_rate * results['placement'])
        
        # Component 3: Performance Trend
        trend_scores = []
        for idx, row in score_matrix.iterrows() if score_cols else []:
            valid_scores = [(i+1, score) for i, score in enumerate(row) if not pd.isna(score)]
            if len(valid_scores) > 1:
                weeks, scores = zip(*valid_scores)
                correlation = np.corrcoef(weeks, scores)[0, 1]
                # Convert correlation (-1 to 1) to 0-100 scale
                trend_score = (correlation + 1) * 50 if not np.isnan(correlation) else 50
                trend_scores.append(trend_score)
            else:
                trend_scores.append(50)  # Neutral score
        
        results['trend_component'] = trend_scores if trend_scores else 50
        
        # Component 4: Consistency Bonus
        if score_cols:
            score_std = score_matrix.std(axis=1)
            # Lower std = higher consistency, convert to 0-100 scale
            # Normalize: consistency = 100 - (std / max_std * 100)
            max_std = score_std.max()
            if max_std > 0:
                results['consistency_component'] = 100 - (score_std / max_std * 100)
            else:
                results['consistency_component'] = 100
            results['consistency_component'] = results['consistency_component'].fillna(100)
        else:
            results['consistency_component'] = 100
        
        # Calculate weighted hybrid score
        weights = hybrid_system['components']
        results['hybrid_score'] = (
            results['judge_component'] * weights['judge_score']['weight'] +
            results['audience_component'] * weights['audience_vote']['weight'] +
            results['trend_component'] * weights['performance_trend']['weight'] +
            results['consistency_component'] * weights['consistency_bonus']['weight']
        )
        
        # Calculate hybrid ranking
        results['hybrid_rank'] = results['hybrid_score'].rank(method='min', ascending=False)
        
        print("\n=== Hybrid Score Calculation Results ===")
        print(f"Average hybrid score: {results['hybrid_score'].mean():.2f}")
        print(f"Score range: {results['hybrid_score'].min():.2f} - {results['hybrid_score'].max():.2f}")
        
        return results
    
    def compare_with_current_system(self, hybrid_results: pd.DataFrame) -> Dict:
        """
        Compare hybrid system with current system
        
        Args:
            hybrid_results: DataFrame with hybrid scores
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate correlation between hybrid rank and actual placement
        hybrid_correlation = hybrid_results[['hybrid_rank', 'placement']].corr().iloc[0, 1]
        
        # Identify ranking changes
        hybrid_results['rank_change'] = hybrid_results['placement'] - hybrid_results['hybrid_rank']
        
        # Count significant changes (more than 2 positions)
        significant_changes = len(hybrid_results[abs(hybrid_results['rank_change']) > 2])
        
        # Calculate average rank change
        avg_rank_change = abs(hybrid_results['rank_change']).mean()
        
        # Winners comparison
        current_winners = hybrid_results[hybrid_results['placement'] == 1][
            ['celebrity_name', 'season', 'placement', 'hybrid_rank', 'hybrid_score']
        ]
        
        comparison = {
            'correlation_with_current': hybrid_correlation,
            'significant_rank_changes': significant_changes,
            'avg_rank_change': avg_rank_change,
            'stability_score': 1 - (significant_changes / len(hybrid_results)),
            'winners_analysis': current_winners
        }
        
        print("\n=== Hybrid System Comparison ===")
        print(f"Correlation with current system: {hybrid_correlation:.4f}")
        print(f"Significant rank changes (>2 positions): {significant_changes}")
        print(f"Average rank change: {avg_rank_change:.2f} positions")
        print(f"System stability score: {comparison['stability_score']:.4f}")
        
        print("\n=== Winners Under Hybrid System ===")
        if len(current_winners) > 0:
            print(current_winners.to_string(index=False))
        
        return comparison
    
    def evaluate_fairness_improvements(self, hybrid_results: pd.DataFrame) -> Dict:
        """
        Evaluate fairness improvements of hybrid system
        
        Args:
            hybrid_results: DataFrame with hybrid scores
            
        Returns:
            Dictionary with fairness metrics
        """
        fairness_metrics = {}
        
        # Metric 1: Score distribution (should be more spread out)
        current_placement_cv = self.data['placement'].std() / self.data['placement'].mean()
        hybrid_score_cv = hybrid_results['hybrid_score'].std() / hybrid_results['hybrid_score'].mean()
        fairness_metrics['discrimination_power'] = {
            'current_cv': current_placement_cv,
            'hybrid_cv': hybrid_score_cv,
            'improvement': 'Better' if hybrid_score_cv > current_placement_cv else 'Worse'
        }
        
        # Metric 2: Variance explained by multiple factors
        # Higher is better (not dominated by single factor)
        component_variance = hybrid_results[[
            'judge_component', 'audience_component', 
            'trend_component', 'consistency_component'
        ]].var()
        
        fairness_metrics['factor_balance'] = {
            'component_variances': component_variance.to_dict(),
            'balance_score': 1 / (component_variance.std() + 1)  # Lower std = more balanced
        }
        
        # Metric 3: Consistency with judge scores and placement
        judge_consistency = hybrid_results[['hybrid_score', 'judge_component']].corr().iloc[0, 1]
        fairness_metrics['judge_consistency'] = judge_consistency
        
        # Metric 4: Reward for improvement
        improvement_correlation = hybrid_results[['trend_component', 'hybrid_rank']].corr().iloc[0, 1]
        fairness_metrics['improvement_reward'] = abs(improvement_correlation)
        
        print("\n=== Fairness Improvement Evaluation ===")
        print(f"\nDiscrimination Power:")
        print(f"  Current CV: {current_placement_cv:.4f}")
        print(f"  Hybrid CV: {hybrid_score_cv:.4f}")
        print(f"  Assessment: {fairness_metrics['discrimination_power']['improvement']}")
        
        print(f"\nFactor Balance Score: {fairness_metrics['factor_balance']['balance_score']:.4f}")
        print(f"Judge Consistency: {judge_consistency:.4f}")
        print(f"Improvement Reward: {fairness_metrics['improvement_reward']:.4f}")
        
        return fairness_metrics
    
    def simulate_enhanced_viewership(self) -> Dict:
        """
        Simulate potential viewership improvements
        
        Returns:
            Dictionary with viewership projections
        """
        # Estimated improvements based on system features
        improvements = {
            'transparency_boost': {
                'factor': 1.15,  # 15% increase from transparency
                'rationale': 'Viewers trust system more with visible vote counts'
            },
            'engagement_boost': {
                'factor': 1.20,  # 20% increase from engagement
                'rationale': 'Multiple voting factors increase viewer participation'
            },
            'fairness_perception': {
                'factor': 1.10,  # 10% increase from fairness
                'rationale': 'Balanced system reduces controversy, improves satisfaction'
            },
            'total_projection': {
                'factor': 1.15 * 1.20 * 1.10,
                'increase_percentage': (1.15 * 1.20 * 1.10 - 1) * 100
            }
        }
        
        print("\n=== Enhanced Viewership Projections ===")
        for improvement, details in improvements.items():
            print(f"\n{improvement}:")
            print(f"  Factor: {details['factor']:.2f}x")
            if 'rationale' in details:
                print(f"  Rationale: {details['rationale']}")
            if 'increase_percentage' in details:
                print(f"  Projected Total Increase: {details['increase_percentage']:.1f}%")
        
        return improvements


def run_task4_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete Task 4 analysis
    
    Args:
        data: Preprocessed DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("TASK 4: PROPOSE IMPROVED VOTING SYSTEM")
    print("="*60)
    
    # Create system designer
    designer = ImprovedVotingSystem(data)
    
    # Analyze current system
    current_issues = designer.current_system_analysis()
    
    # Design hybrid system
    hybrid_system = designer.design_hybrid_system()
    
    # Calculate hybrid scores
    hybrid_results = designer.calculate_hybrid_scores(hybrid_system)
    
    # Compare with current system
    comparison = designer.compare_with_current_system(hybrid_results)
    
    # Evaluate fairness improvements
    fairness_eval = designer.evaluate_fairness_improvements(hybrid_results)
    
    # Simulate viewership improvements
    viewership_proj = designer.simulate_enhanced_viewership()
    
    return {
        'designer': designer,
        'current_issues': current_issues,
        'hybrid_system': hybrid_system,
        'hybrid_results': hybrid_results,
        'comparison': comparison,
        'fairness_evaluation': fairness_eval,
        'viewership_projections': viewership_proj
    }


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor('2026_MCM_Problem_C_Data.csv')
    data = preprocessor.preprocess_all()
    
    # Run Task 4 analysis
    task4_results = run_task4_analysis(data)
    
    # Save results
    hybrid_results = task4_results['hybrid_results'][[
        'celebrity_name', 'season', 'placement', 
        'hybrid_score', 'hybrid_rank', 'rank_change'
    ]].sort_values('hybrid_score', ascending=False)
    
    hybrid_results.to_csv('task4_hybrid_system_results.csv', index=False)
    print("\nHybrid system results saved to: task4_hybrid_system_results.csv")
