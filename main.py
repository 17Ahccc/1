"""
主运行脚本 (Main Runner Script)
执行完整的MCM分析流程

使用方法:
    python main.py --all                    # 运行所有分析
    python main.py --preprocess             # 仅数据预处理
    python main.py --eda                    # 仅探索性数据分析
    python main.py --statistical            # 仅统计建模
    python main.py --predict                # 仅预测建模
    python main.py --evaluate               # 仅模型评估
    python main.py --visualize              # 仅可视化
"""

import sys
import argparse
import time
from src.data_preprocessing import DWTSDataPreprocessor
from src.exploratory_analysis import EDAnalyzer
from src.statistical_models import StatisticalModeler
from src.prediction_models import PredictionModeler
from src.model_evaluation import ModelEvaluator
from src.visualization import Visualizer


def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    MCM 2026 Problem C: Dancing with the Stars Analysis       ║
    ║    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
    ║                                                               ║
    ║    Mathematical Contest in Modeling                          ║
    ║    Statistical Analysis & Prediction Framework               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_preprocessing():
    """运行数据预处理"""
    print("\n" + "=" * 80)
    print("Step 1: 数据预处理 (Data Preprocessing)")
    print("=" * 80)
    
    start_time = time.time()
    
    preprocessor = DWTSDataPreprocessor()
    preprocessor.load_data()
    preprocessor.explore_data_structure()
    processed_data = preprocessor.process()
    preprocessor.save_processed_data()
    
    stats = preprocessor.get_summary_statistics()
    print("\n数据摘要:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    elapsed = time.time() - start_time
    print(f"\n✓ 数据预处理完成! 耗时: {elapsed:.2f}秒")


def run_eda():
    """运行探索性数据分析"""
    print("\n" + "=" * 80)
    print("Step 2: 探索性数据分析 (Exploratory Data Analysis)")
    print("=" * 80)
    
    start_time = time.time()
    
    analyzer = EDAnalyzer()
    analyzer.run_full_analysis()
    
    elapsed = time.time() - start_time
    print(f"\n✓ 探索性数据分析完成! 耗时: {elapsed:.2f}秒")


def run_statistical_models():
    """运行统计建模"""
    print("\n" + "=" * 80)
    print("Step 3: 统计建模 (Statistical Modeling)")
    print("=" * 80)
    
    start_time = time.time()
    
    modeler = StatisticalModeler()
    modeler.run_all_models()
    
    elapsed = time.time() - start_time
    print(f"\n✓ 统计建模完成! 耗时: {elapsed:.2f}秒")


def run_prediction_models():
    """运行预测建模"""
    print("\n" + "=" * 80)
    print("Step 4: 预测建模 (Prediction Modeling)")
    print("=" * 80)
    
    start_time = time.time()
    
    modeler = PredictionModeler()
    modeler.run_all_models()
    
    elapsed = time.time() - start_time
    print(f"\n✓ 预测建模完成! 耗时: {elapsed:.2f}秒")


def run_evaluation():
    """运行模型评估"""
    print("\n" + "=" * 80)
    print("Step 5: 模型评估 (Model Evaluation)")
    print("=" * 80)
    
    start_time = time.time()
    
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()
    
    elapsed = time.time() - start_time
    print(f"\n✓ 模型评估完成! 耗时: {elapsed:.2f}秒")


def run_visualization():
    """运行可视化"""
    print("\n" + "=" * 80)
    print("Step 6: 数据可视化 (Data Visualization)")
    print("=" * 80)
    
    start_time = time.time()
    
    visualizer = Visualizer()
    visualizer.generate_all_visualizations()
    
    elapsed = time.time() - start_time
    print(f"\n✓ 可视化生成完成! 耗时: {elapsed:.2f}秒")


def run_all():
    """运行完整分析流程"""
    print_banner()
    
    total_start = time.time()
    
    # 执行所有步骤
    run_preprocessing()
    run_eda()
    run_statistical_models()
    run_prediction_models()
    run_evaluation()
    run_visualization()
    
    # 打印总结
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("✓ 完整分析流程执行完成!")
    print("=" * 80)
    print(f"\n总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
    
    print("\n生成的文件:")
    print("  • 数据文件: data/processed_data.csv")
    print("  • 图表文件: results/figures/*.png")
    print("  • 交互式图表: results/figures/interactive_dashboard.html")
    
    print("\n建议的下一步:")
    print("  1. 查看 results/figures/ 目录中的所有可视化结果")
    print("  2. 查看交互式仪表板以深入探索数据")
    print("  3. 使用生成的图表和统计结果撰写MCM论文")
    print("  4. 根据需要调整模型参数以优化性能")
    
    print("\n祝你在MCM竞赛中取得优异成绩! 🏆")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MCM 2026 Problem C Analysis Framework'
    )
    
    parser.add_argument('--all', action='store_true',
                       help='运行完整分析流程')
    parser.add_argument('--preprocess', action='store_true',
                       help='仅运行数据预处理')
    parser.add_argument('--eda', action='store_true',
                       help='仅运行探索性数据分析')
    parser.add_argument('--statistical', action='store_true',
                       help='仅运行统计建模')
    parser.add_argument('--predict', action='store_true',
                       help='仅运行预测建模')
    parser.add_argument('--evaluate', action='store_true',
                       help='仅运行模型评估')
    parser.add_argument('--visualize', action='store_true',
                       help='仅运行可视化')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认运行所有
    if not any(vars(args).values()):
        args.all = True
    
    try:
        if args.all:
            run_all()
        else:
            print_banner()
            if args.preprocess:
                run_preprocessing()
            if args.eda:
                run_eda()
            if args.statistical:
                run_statistical_models()
            if args.predict:
                run_prediction_models()
            if args.evaluate:
                run_evaluation()
            if args.visualize:
                run_visualization()
    
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
