"""
Problem 3: Fairness Analysis using ANOVA and Linear Mixed Models

This module:
- Performs ANOVA to test for systematic biases
- Fits Linear Mixed Models (LMM) for random effects
- Analyzes fairness based on residuals
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import seaborn as sns
import os


class FairnessAnalyzer:
    """Analyzes fairness using ANOVA and Linear Mixed Models."""
    
    def __init__(self):
        """Initialize the fairness analyzer."""
        self.anova_results = {}
        self.lmm_model = None
        self.lmm_results = None
    
    def prepare_fairness_data(self, df_with_residuals):
        """
        Prepare data for fairness analysis.
        
        Args:
            df_with_residuals (pd.DataFrame): Dataset with residuals
            
        Returns:
            pd.DataFrame: Prepared data for analysis
        """
        df = df_with_residuals.copy()
        
        # Extract categorical variables from one-hot encoded columns
        # Extract industry
        industry_cols = [col for col in df.columns if col.startswith('celebrity_industry_')]
        if industry_cols:
            df['industry'] = 'Other'
            for col in industry_cols:
                industry_name = col.replace('celebrity_industry_', '')
                df.loc[df[col] == 1, 'industry'] = industry_name
        else:
            df['industry'] = 'Unknown'
        
        # Use season as is
        df['season_cat'] = df['season'].astype(str)
        
        # Create judge_id (for LMM random effects)
        # In this dataset, we'll use week as a proxy for judge variability
        df['judge_id'] = df['week'].astype(str) if 'week' in df.columns else '1'
        
        # Filter out rows with missing residuals
        df = df[~df['residual'].isna()]
        
        print(f"Prepared {len(df)} records for fairness analysis")
        print(f"Industries: {df['industry'].unique()[:10]}")
        print(f"Seasons: {df['season_cat'].unique()[:10]}")
        
        return df
    
    def perform_anova_industry(self, df):
        """
        Perform ANOVA to test if industry affects residuals.
        
        Args:
            df (pd.DataFrame): Data with residuals and industry
            
        Returns:
            dict: ANOVA results
        """
        print("\n=== ANOVA: Industry Effect on Residuals ===")
        
        # Group residuals by industry
        groups = []
        industries = df['industry'].unique()
        
        for industry in industries:
            group_residuals = df[df['industry'] == industry]['residual'].values
            if len(group_residuals) > 0:
                groups.append(group_residuals)
        
        # Perform one-way ANOVA
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            
            result = {
                'variable': 'industry',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'num_groups': len(groups)
            }
            
            print(f"F-statistic: {f_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Significant (α=0.05): {result['significant']}")
            
            if result['significant']:
                print("→ Industry has a SIGNIFICANT effect on residuals (potential bias)")
            else:
                print("→ Industry has NO significant effect on residuals")
        else:
            result = {'error': 'Not enough groups for ANOVA'}
            print("Error: Not enough groups for ANOVA")
        
        self.anova_results['industry'] = result
        print("=== ANOVA Complete ===\n")
        
        return result
    
    def perform_anova_season(self, df):
        """
        Perform ANOVA to test if season affects residuals.
        
        Args:
            df (pd.DataFrame): Data with residuals and season
            
        Returns:
            dict: ANOVA results
        """
        print("\n=== ANOVA: Season Effect on Residuals ===")
        
        # Group residuals by season
        groups = []
        seasons = df['season_cat'].unique()
        
        for season in seasons:
            group_residuals = df[df['season_cat'] == season]['residual'].values
            if len(group_residuals) > 0:
                groups.append(group_residuals)
        
        # Perform one-way ANOVA
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            
            result = {
                'variable': 'season',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'num_groups': len(groups)
            }
            
            print(f"F-statistic: {f_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Significant (α=0.05): {result['significant']}")
            
            if result['significant']:
                print("→ Season has a SIGNIFICANT effect on residuals (potential bias)")
            else:
                print("→ Season has NO significant effect on residuals")
        else:
            result = {'error': 'Not enough groups for ANOVA'}
            print("Error: Not enough groups for ANOVA")
        
        self.anova_results['season'] = result
        print("=== ANOVA Complete ===\n")
        
        return result
    
    def fit_linear_mixed_model(self, df):
        """
        Fit Linear Mixed Model with fixed and random effects.
        
        Args:
            df (pd.DataFrame): Data with residuals, industry, season, judge_id
            
        Returns:
            Mixed model results
        """
        print("\n=== Fitting Linear Mixed Model ===")
        print("Model: residual ~ industry + season + (1|judge_id)")
        
        try:
            # Prepare data - ensure we have enough variation
            # Filter to industries with sufficient samples
            industry_counts = df['industry'].value_counts()
            valid_industries = industry_counts[industry_counts >= 5].index
            df_filtered = df[df['industry'].isin(valid_industries)].copy()
            
            if len(df_filtered) < 20:
                print("Warning: Not enough data for LMM. Using simplified analysis.")
                return None
            
            # Fit the mixed model
            # Formula: residual ~ industry + season + (1|judge_id)
            formula = "residual ~ C(industry) + C(season_cat)"
            
            self.lmm_model = mixedlm(formula, df_filtered, groups=df_filtered["judge_id"])
            self.lmm_results = self.lmm_model.fit(method='lbfgs')
            
            print("\n" + "="*60)
            print(self.lmm_results.summary())
            print("="*60)
            
            # Interpret results
            print("\n=== Interpretation ===")
            
            # Check fixed effects significance
            pvalues = self.lmm_results.pvalues
            significant_effects = pvalues[pvalues < 0.05]
            
            if len(significant_effects) > 0:
                print("Significant fixed effects found:")
                for effect, pval in significant_effects.items():
                    print(f"  - {effect}: p={pval:.4f}")
                print("→ This indicates potential SYSTEMATIC BIAS")
            else:
                print("No significant fixed effects found")
                print("→ This suggests FAIR judging across groups")
            
            # Check random effects (judge subjectivity)
            re_var = self.lmm_results.cov_re
            print(f"\nRandom effect variance (judge subjectivity): {re_var.iloc[0, 0]:.4f}")
            
            if re_var.iloc[0, 0] > 0.5:
                print("→ High judge subjectivity detected")
            else:
                print("→ Low judge subjectivity (consistent judging)")
            
            print("=== LMM Complete ===\n")
            
        except Exception as e:
            print(f"Error fitting LMM: {e}")
            print("Using simplified analysis instead...")
            self.lmm_results = None
        
        return self.lmm_results
    
    def plot_fairness_analysis(self, df, output_dir='visualizations'):
        """
        Create visualizations for fairness analysis.
        
        Args:
            df (pd.DataFrame): Data with residuals
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Residuals by industry
        if 'industry' in df.columns:
            industry_data = df.groupby('industry')['residual'].apply(list)
            if len(industry_data) > 1:
                axes[0, 0].boxplot([data for data in industry_data.values if len(data) > 0],
                                  labels=[ind[:15] for ind, data in industry_data.items() if len(data) > 0])
                axes[0, 0].set_xlabel('Industry')
                axes[0, 0].set_ylabel('Residual')
                axes[0, 0].set_title('Residuals by Industry')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals by season
        if 'season_cat' in df.columns:
            season_data = df.groupby('season_cat')['residual'].apply(list)
            if len(season_data) > 1:
                axes[0, 1].boxplot([data for data in season_data.values if len(data) > 0],
                                  labels=list(season_data.keys()))
                axes[0, 1].set_xlabel('Season')
                axes[0, 1].set_ylabel('Residual')
                axes[0, 1].set_title('Residuals by Season')
                axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        axes[1, 0].hist(df['residual'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q plot for normality
        stats.probplot(df['residual'].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'fairness_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved fairness analysis plots to {output_path}")
        plt.close()
    
    def generate_fairness_report(self, output_dir='reports'):
        """
        Generate a text report of fairness analysis.
        
        Args:
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'fairness_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FAIRNESS ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # ANOVA results
            f.write("1. ANOVA RESULTS\n")
            f.write("-" * 70 + "\n\n")
            
            for var, result in self.anova_results.items():
                if 'error' not in result:
                    f.write(f"{var.upper()}:\n")
                    f.write(f"  F-statistic: {result['f_statistic']:.4f}\n")
                    f.write(f"  P-value: {result['p_value']:.4f}\n")
                    f.write(f"  Significant: {result['significant']}\n")
                    if result['significant']:
                        f.write(f"  → POTENTIAL BIAS DETECTED\n")
                    else:
                        f.write(f"  → No significant bias\n")
                    f.write("\n")
            
            # LMM results
            f.write("\n2. LINEAR MIXED MODEL RESULTS\n")
            f.write("-" * 70 + "\n\n")
            
            if self.lmm_results is not None:
                f.write("Model: residual ~ industry + season + (1|judge_id)\n\n")
                f.write(str(self.lmm_results.summary()))
                f.write("\n\n")
            else:
                f.write("LMM could not be fitted. See console output for details.\n\n")
            
            # Conclusions
            f.write("\n3. CONCLUSIONS\n")
            f.write("-" * 70 + "\n\n")
            
            significant_biases = [var for var, res in self.anova_results.items() 
                                 if 'error' not in res and res['significant']]
            
            if significant_biases:
                f.write("SYSTEMATIC BIASES DETECTED:\n")
                for var in significant_biases:
                    f.write(f"  - {var.upper()}\n")
                f.write("\nRecommendation: Review judging criteria and procedures.\n")
            else:
                f.write("NO SYSTEMATIC BIASES DETECTED\n")
                f.write("\nThe judging appears to be fair across groups.\n")
        
        print(f"Fairness analysis report saved to {report_path}")


if __name__ == "__main__":
    print("Fairness Analysis Module")
    print("This module should be run as part of the main pipeline.")
