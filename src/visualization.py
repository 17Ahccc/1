"""
可视化模块 (Visualization Module)
生成高质量的数据可视化图表用于论文

包括:
1. 评分趋势可视化
2. 特征重要性可视化
3. 相关性分析可视化
4. 预测结果可视化
5. 交互式可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """可视化器"""
    
    def __init__(self, data_path: str = 'data/processed_data.csv'):
        """
        初始化可视化器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.results_dir = 'results/figures'
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成: {self.data.shape}")
        return self.data
    
    def plot_score_evolution_by_contestant(self, top_n=10):
        """
        绘制顶级选手的评分演变图
        
        Args:
            top_n: 显示前N名选手
        """
        print(f"\n生成选手评分演变图（Top {top_n}）...")
        
        if self.data is None:
            self.load_data()
        
        # 选择排名前N的选手
        top_contestants = self.data.nsmallest(top_n, 'placement')
        
        # 获取评分列
        score_cols = [col for col in self.data.columns if 'week' in col and 'score' in col and 'judge1' in col]
        
        plt.figure(figsize=(14, 8))
        
        for idx, row in top_contestants.iterrows():
            weeks = []
            scores = []
            
            for col in score_cols:
                week_num = int(col.split('_')[0].replace('week', ''))
                score = row[col]
                
                if pd.notna(score) and score > 0:
                    weeks.append(week_num)
                    scores.append(score)
            
            if len(weeks) > 0:
                plt.plot(weeks, scores, marker='o', linewidth=2, 
                        label=f"{row['celebrity_name']} (Rank {int(row['placement'])})",
                        alpha=0.7)
        
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Score (Judge 1)', fontsize=12)
        plt.title(f'Score Evolution for Top {top_n} Contestants', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/score_evolution_top_{top_n}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"评分演变图已保存")
        plt.close()
    
    def plot_age_vs_performance(self):
        """绘制年龄与表现的关系"""
        print("\n生成年龄-表现关系图...")
        
        if self.data is None:
            self.load_data()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 年龄 vs 排名
        axes[0].scatter(self.data['celebrity_age_during_season'], 
                       self.data['placement'], 
                       alpha=0.6, s=50, c=self.data['avg_score'], 
                       cmap='viridis')
        axes[0].set_xlabel('Age', fontsize=12)
        axes[0].set_ylabel('Placement (lower is better)', fontsize=12)
        axes[0].set_title('Age vs Final Placement', fontsize=14)
        axes[0].invert_yaxis()
        
        # 添加趋势线
        z = np.polyfit(self.data['celebrity_age_during_season'].dropna(), 
                      self.data['placement'].dropna(), 1)
        p = np.poly1d(z)
        axes[0].plot(self.data['celebrity_age_during_season'].dropna().sort_values(), 
                    p(self.data['celebrity_age_during_season'].dropna().sort_values()), 
                    "r--", alpha=0.8, linewidth=2)
        
        # 年龄 vs 平均分
        axes[1].scatter(self.data['celebrity_age_during_season'], 
                       self.data['avg_score'], 
                       alpha=0.6, s=50, c=self.data['num_weeks_participated'], 
                       cmap='plasma')
        axes[1].set_xlabel('Age', fontsize=12)
        axes[1].set_ylabel('Average Score', fontsize=12)
        axes[1].set_title('Age vs Average Score', fontsize=14)
        
        # 添加颜色条
        cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
        cbar.set_label('Weeks Participated', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/age_vs_performance.png', 
                   dpi=300, bbox_inches='tight')
        print(f"年龄-表现关系图已保存")
        plt.close()
    
    def plot_industry_analysis(self):
        """绘制行业分析图"""
        print("\n生成行业分析图...")
        
        if self.data is None:
            self.load_data()
        
        if 'celebrity_industry' not in self.data.columns:
            print("缺少行业信息，跳过此可视化")
            return
        
        # 统计各行业的表现
        industry_stats = self.data.groupby('celebrity_industry').agg({
            'avg_score': 'mean',
            'placement': 'mean',
            'num_weeks_participated': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'count'})
        
        # 只保留样本量>=5的行业
        industry_stats = industry_stats[industry_stats['count'] >= 5].sort_values('avg_score', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Industry Performance Analysis', fontsize=16, y=0.995)
        
        # 1. 各行业平均分
        axes[0, 0].barh(industry_stats.index, industry_stats['avg_score'], color='skyblue')
        axes[0, 0].set_xlabel('Average Score')
        axes[0, 0].set_title('Average Score by Industry')
        axes[0, 0].invert_yaxis()
        
        # 2. 各行业平均排名
        axes[0, 1].barh(industry_stats.index, industry_stats['placement'], color='lightcoral')
        axes[0, 1].set_xlabel('Average Placement (lower is better)')
        axes[0, 1].set_title('Average Placement by Industry')
        axes[0, 1].invert_yaxis()
        
        # 3. 各行业参赛周数
        axes[1, 0].barh(industry_stats.index, industry_stats['num_weeks_participated'], color='lightgreen')
        axes[1, 0].set_xlabel('Average Weeks Participated')
        axes[1, 0].set_title('Participation Duration by Industry')
        axes[1, 0].invert_yaxis()
        
        # 4. 各行业样本量
        axes[1, 1].barh(industry_stats.index, industry_stats['count'], color='gold')
        axes[1, 1].set_xlabel('Number of Contestants')
        axes[1, 1].set_title('Sample Size by Industry')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/industry_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print(f"行业分析图已保存")
        plt.close()
    
    def plot_season_trends(self):
        """绘制赛季趋势图"""
        print("\n生成赛季趋势图...")
        
        if self.data is None:
            self.load_data()
        
        # 统计各赛季的指标
        season_stats = self.data.groupby('season').agg({
            'avg_score': 'mean',
            'celebrity_age_during_season': 'mean',
            'num_weeks_participated': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'contestants'})
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Season-wise Trends Analysis', fontsize=16, y=0.995)
        
        # 1. 平均分趋势
        axes[0, 0].plot(season_stats.index, season_stats['avg_score'], 
                       marker='o', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_xlabel('Season')
        axes[0, 0].set_ylabel('Average Score')
        axes[0, 0].set_title('Average Score Trend Across Seasons')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 平均年龄趋势
        axes[0, 1].plot(season_stats.index, season_stats['celebrity_age_during_season'], 
                       marker='s', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Average Age')
        axes[0, 1].set_title('Average Contestant Age Across Seasons')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 参赛周数趋势
        axes[1, 0].plot(season_stats.index, season_stats['num_weeks_participated'], 
                       marker='^', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Average Weeks')
        axes[1, 0].set_title('Average Participation Duration Across Seasons')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 参赛人数
        axes[1, 1].bar(season_stats.index, season_stats['contestants'], color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Season')
        axes[1, 1].set_ylabel('Number of Contestants')
        axes[1, 1].set_title('Number of Contestants per Season')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/season_trends.png', 
                   dpi=300, bbox_inches='tight')
        print(f"赛季趋势图已保存")
        plt.close()
    
    def plot_score_distribution_by_week(self):
        """绘制各周评分分布"""
        print("\n生成各周评分分布图...")
        
        if self.data is None:
            self.load_data()
        
        # 获取各周的第一个评委的分数
        score_cols = [col for col in self.data.columns if 'judge1_score' in col]
        
        # 提取各周数据
        weekly_data = []
        for col in score_cols:
            week = col.split('_')[0].replace('week', '')
            if week.isdigit():
                scores = self.data[col].dropna()
                scores = scores[scores > 0]  # 排除0分
                for score in scores:
                    weekly_data.append({'Week': int(week), 'Score': score})
        
        weekly_df = pd.DataFrame(weekly_data)
        
        # 绘制小提琴图
        plt.figure(figsize=(16, 6))
        sns.violinplot(data=weekly_df, x='Week', y='Score', palette='Set2')
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Score Distribution by Week (Judge 1)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/score_distribution_by_week.png', 
                   dpi=300, bbox_inches='tight')
        print(f"各周评分分布图已保存")
        plt.close()
    
    def create_interactive_dashboard(self):
        """创建交互式仪表板"""
        print("\n生成交互式仪表板...")
        
        if self.data is None:
            self.load_data()
        
        # 创建散点图：年龄 vs 平均分 vs 排名
        fig = px.scatter(
            self.data,
            x='celebrity_age_during_season',
            y='avg_score',
            size='num_weeks_participated',
            color='placement',
            hover_data=['celebrity_name', 'celebrity_industry', 'season'],
            title='Interactive: Age vs Average Score (sized by weeks, colored by placement)',
            labels={
                'celebrity_age_during_season': 'Age',
                'avg_score': 'Average Score',
                'num_weeks_participated': 'Weeks Participated',
                'placement': 'Final Placement'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=600, width=1000)
        fig.write_html(f'{self.results_dir}/interactive_dashboard.html')
        print(f"交互式仪表板已保存: {self.results_dir}/interactive_dashboard.html")
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        print("\n" + "=" * 80)
        print("开始生成可视化")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 生成各类可视化
        self.plot_score_evolution_by_contestant(top_n=10)
        self.plot_age_vs_performance()
        self.plot_industry_analysis()
        self.plot_season_trends()
        self.plot_score_distribution_by_week()
        self.create_interactive_dashboard()
        
        print("\n" + "=" * 80)
        print("可视化生成完成!")
        print("=" * 80)


def main():
    """主函数"""
    # 创建可视化器实例
    visualizer = Visualizer()
    
    # 生成所有可视化
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
