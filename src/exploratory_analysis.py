"""
探索性数据分析模块 (Exploratory Data Analysis Module)
用于对Dancing with the Stars数据进行全面的统计分析和可视化

功能:
1. 描述性统计分析
2. 分布特征分析
3. 相关性分析
4. 假设检验
5. 趋势分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


class EDAnalyzer:
    """探索性数据分析器"""
    
    def __init__(self, data_path: str = 'data/processed_data.csv'):
        """
        初始化分析器
        
        Args:
            data_path: 处理后的数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.results_dir = 'results/figures'
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成: {self.data.shape}")
        return self.data
    
    def descriptive_statistics(self) -> pd.DataFrame:
        """
        描述性统计分析
        
        Returns:
            统计摘要DataFrame
        """
        print("\n" + "=" * 60)
        print("描述性统计分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 数值型变量的描述统计
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        desc_stats = self.data[numeric_cols].describe()
        
        print("\n数值型变量统计摘要:")
        print(desc_stats)
        
        # 分类变量的频数统计
        print("\n分类变量频数统计:")
        
        if 'celebrity_industry' in self.data.columns:
            print("\n行业分布:")
            print(self.data['celebrity_industry'].value_counts())
        
        if 'results' in self.data.columns:
            print("\n比赛结果分布:")
            print(self.data['results'].value_counts().head(10))
        
        return desc_stats
    
    def distribution_analysis(self):
        """
        分布特征分析
        生成主要变量的分布图
        """
        print("\n" + "=" * 60)
        print("分布特征分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Key Variables Distribution Analysis', fontsize=16, y=0.995)
        
        # 1. 年龄分布
        axes[0, 0].hist(self.data['celebrity_age_during_season'].dropna(), 
                        bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].axvline(self.data['celebrity_age_during_season'].mean(), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # 2. 平均分分布
        axes[0, 1].hist(self.data['avg_score'].dropna(), 
                        bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Average Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Average Score Distribution')
        
        # 3. 参赛周数分布
        axes[0, 2].hist(self.data['num_weeks_participated'].dropna(), 
                        bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Weeks Participated')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Participation Duration Distribution')
        
        # 4. 赛季分布
        season_counts = self.data['season'].value_counts().sort_index()
        axes[1, 0].bar(season_counts.index, season_counts.values, color='gold', alpha=0.7)
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Number of Celebrities')
        axes[1, 0].set_title('Celebrities per Season')
        
        # 5. 行业分布
        if 'celebrity_industry' in self.data.columns:
            industry_counts = self.data['celebrity_industry'].value_counts().head(10)
            axes[1, 1].barh(range(len(industry_counts)), industry_counts.values, color='purple', alpha=0.7)
            axes[1, 1].set_yticks(range(len(industry_counts)))
            axes[1, 1].set_yticklabels(industry_counts.index, fontsize=8)
            axes[1, 1].set_xlabel('Count')
            axes[1, 1].set_title('Top 10 Industries')
        
        # 6. 分数标准差分布
        axes[1, 2].hist(self.data['std_score'].dropna(), 
                        bins=25, color='orange', edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('Score Standard Deviation')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Score Variability Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"分布分析图已保存至: {self.results_dir}/distribution_analysis.png")
        plt.close()
    
    def correlation_analysis(self):
        """
        相关性分析
        生成相关系数矩阵热图
        """
        print("\n" + "=" * 60)
        print("相关性分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 选择关键数值型变量
        key_vars = [
            'celebrity_age_during_season',
            'placement',
            'avg_score',
            'max_score',
            'min_score',
            'std_score',
            'num_weeks_participated',
            'score_trend'
        ]
        
        # 过滤存在的列
        available_vars = [var for var in key_vars if var in self.data.columns]
        
        # 计算相关系数矩阵
        corr_matrix = self.data[available_vars].corr()
        
        print("\n相关系数矩阵:")
        print(corr_matrix)
        
        # 绘制热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Key Variables', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"相关性热图已保存至: {self.results_dir}/correlation_matrix.png")
        plt.close()
        
        # 识别强相关关系
        print("\n强相关关系 (|r| > 0.5):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    def hypothesis_testing(self):
        """
        假设检验
        测试不同组别之间的显著性差异
        """
        print("\n" + "=" * 60)
        print("假设检验")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 测试1: 不同行业的平均分是否有显著差异（ANOVA）
        if 'celebrity_industry' in self.data.columns:
            print("\n测试1: 不同行业平均分差异（ANOVA）")
            
            # 获取主要行业
            top_industries = self.data['celebrity_industry'].value_counts().head(5).index
            industry_scores = [
                self.data[self.data['celebrity_industry'] == ind]['avg_score'].dropna()
                for ind in top_industries
            ]
            
            f_stat, p_value = stats.f_oneway(*industry_scores)
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  结论: 不同行业平均分存在显著差异 (p < 0.05)")
            else:
                print("  结论: 不同行业平均分无显著差异 (p >= 0.05)")
        
        # 测试2: 年龄与最终排名的相关性
        print("\n测试2: 年龄与最终排名的相关性")
        age_placement_data = self.data[['celebrity_age_during_season', 'placement']].dropna()
        
        if len(age_placement_data) > 0:
            corr, p_value = stats.pearsonr(
                age_placement_data['celebrity_age_during_season'],
                age_placement_data['placement']
            )
            print(f"  Pearson相关系数: {corr:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  结论: 年龄与排名存在显著相关性 (p < 0.05)")
            else:
                print("  结论: 年龄与排名无显著相关性 (p >= 0.05)")
        
        # 测试3: 分数标准差的正态性检验
        print("\n测试3: 分数标准差的正态性检验（Shapiro-Wilk）")
        std_scores = self.data['std_score'].dropna()
        
        if len(std_scores) > 0:
            stat, p_value = stats.shapiro(std_scores[:5000])  # 限制样本量
            print(f"  W-statistic: {stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  结论: 分数标准差不服从正态分布 (p < 0.05)")
            else:
                print("  结论: 分数标准差服从正态分布 (p >= 0.05)")
    
    def score_trend_analysis(self):
        """
        评分趋势分析
        分析选手在比赛过程中的评分变化趋势
        """
        print("\n" + "=" * 60)
        print("评分趋势分析")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # 获取评分列
        score_cols = [col for col in self.data.columns if 'week' in col and 'score' in col]
        
        # 按周提取评分
        weekly_scores = {}
        for col in score_cols:
            week_num = col.split('_')[0].replace('week', '')
            if week_num.isdigit():
                week_num = int(week_num)
                if week_num not in weekly_scores:
                    weekly_scores[week_num] = []
                weekly_scores[week_num].extend(self.data[col].dropna().values)
        
        # 计算每周平均分
        weeks = sorted(weekly_scores.keys())
        avg_scores_by_week = [np.mean(weekly_scores[w]) for w in weeks]
        
        # 绘图
        plt.figure(figsize=(14, 6))
        plt.plot(weeks, avg_scores_by_week, marker='o', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.title('Average Score Trend Across Weeks', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/score_trend_by_week.png', dpi=300, bbox_inches='tight')
        print(f"评分趋势图已保存至: {self.results_dir}/score_trend_by_week.png")
        plt.close()
    
    def run_full_analysis(self):
        """运行完整的EDA流程"""
        print("\n" + "=" * 80)
        print("开始探索性数据分析")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 执行各项分析
        self.descriptive_statistics()
        self.distribution_analysis()
        self.correlation_analysis()
        self.hypothesis_testing()
        self.score_trend_analysis()
        
        print("\n" + "=" * 80)
        print("探索性数据分析完成!")
        print("=" * 80)


def main():
    """主函数"""
    # 创建分析器实例
    analyzer = EDAnalyzer()
    
    # 运行完整分析
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
