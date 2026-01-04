# -*- coding: utf-8 -*-
"""
实验结果展示脚本
展示对抗性改写实验的结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载实验结果"""
    try:
        results_df = pd.read_csv('attack_results.csv')
        return results_df
    except FileNotFoundError:
        print("未找到attack_results.csv文件")
        return None

def plot_attack_comparison(results_df):
    """绘制攻击效果对比图"""
    if results_df is None:
        return

    # 设置图表风格
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('对抗性改写攻击效果对比', fontsize=16, fontweight='bold')

    attack_types = ['synonym', 'restructure', 'obfuscation', 'combined']
    attack_names = ['同义词替换', '句式重构', '模糊化表达', '组合攻击']

    for i, (attack_type, attack_name) in enumerate(zip(attack_types, attack_names)):
        ax = axes[i//2, i%2]

        attack_data = results_df[results_df['attack_type'] == attack_type]

        models = attack_data['model']
        accuracy_drop = attack_data['accuracy_drop'] * 100  # 转换为百分比
        attack_success = attack_data['attack_success_rate'] * 100

        x = range(len(models))
        width = 0.35

        bars1 = ax.bar([pos - width/2 for pos in x], accuracy_drop,
                      width, label='准确率下降', color='lightcoral', alpha=0.7)
        bars2 = ax.bar([pos + width/2 for pos in x], attack_success,
                      width, label='攻击成功率', color='skyblue', alpha=0.7)

        ax.set_title(f'{attack_name}攻击效果', fontsize=12, fontweight='bold')
        ax.set_xlabel('模型类型')
        ax.set_ylabel('百分比 (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   '.1f', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   '.1f', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('attack_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results_df):
    """打印结果汇总表"""
    if results_df is None:
        return

    print("\n" + "="*80)
    print("对抗性改写实验结果汇总")
    print("="*80)

    # 按攻击类型分组显示
    for attack_type in results_df['attack_type'].unique():
        print(f"\n{attack_type.upper()} 攻击策略:")
        print("-" * 40)

        attack_data = results_df[results_df['attack_type'] == attack_type]

        for _, row in attack_data.iterrows():
            model = row['model']
            orig_acc = row['original_accuracy'] * 100
            adv_acc = row['adversarial_accuracy'] * 100
            acc_drop = row['accuracy_drop'] * 100
            success_rate = row['attack_success_rate'] * 100

            print("25"
                  "6.2f"
                  "6.2f"
                  "6.2f"
                  "6.2f")

def analyze_findings(results_df):
    """分析实验发现"""
    if results_df is None:
        return

    print("\n" + "="*80)
    print("实验发现分析")
    print("="*80)

    # 分析最有效的攻击类型
    best_attacks = results_df.loc[results_df.groupby('model')['accuracy_drop'].idxmax()]

    print("\n各模型最有效的攻击策略:")
    for _, row in best_attacks.iterrows():
        print("15"
              "15"
              ".3f")

    # 分析模型鲁棒性
    model_robustness = results_df.groupby('model')['accuracy_drop'].mean().sort_values()

    print("\n模型鲁棒性排名 (平均准确率下降从低到高):")
    for model, avg_drop in model_robustness.items():
        robustness = "高" if avg_drop < 0.001 else "中" if avg_drop < 0.01 else "低"
        print("15"
              "6.4f"
              "8")

def show_sample_examples():
    """展示改写示例"""
    print("\n" + "="*80)
    print("改写示例展示")
    print("="*80)

    examples = [
        {
            'original': "您好，我是银行工作人员，需要验证您的账户信息",
            'synonym': "您好，我是银行员工，需要确认您的账户资料",
            'restructure': "您好，需要验证您的账户信息，我是银行工作人员",
            'obfuscation': "您好，我是银行工作人员，需要核实您的账户详情"
        },
        {
            'original': "这笔投资收益很高，有30%的回报率",
            'synonym': "这个投资收益很高，有30%的回报率",
            'restructure': "有30%的回报率，这笔投资收益很高",
            'obfuscation': "这个投资机会不错，收益相当可观"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print(f"原始: {example['original']}")
        print(f"同义词替换: {example['synonym']}")
        print(f"句式重构: {example['restructure']}")
        print(f"模糊化表达: {example['obfuscation']}")

def main():
    """主函数"""
    print("对抗性改写实验结果展示")
    print("=" * 50)

    # 加载结果
    results_df = load_results()

    if results_df is not None:
        # 显示汇总表
        print_summary_table(results_df)

        # 分析发现
        analyze_findings(results_df)

        # 展示示例
        show_sample_examples()

        # 绘制图表
        try:
            plot_attack_comparison(results_df)
            print("\n对比图已保存为: attack_comparison.png")
        except Exception as e:
            print(f"\n绘图时出错: {e}")

    else:
        print("无法加载实验结果，请确保attack_results.csv文件存在")

if __name__ == "__main__":
    main()
