# -*- coding: utf-8 -*-
"""
实验二：对话行为与交互策略建模
实现对话行为识别与联合分析

功能：
1. 数据准备：加载对话数据，按轮次划分，提取left端句子
2. 对话行为类别自动识别：使用规则-based方法为句子生成行为标签
3. 联合分析：行为×策略×诈骗的关系分析
4. 解释建模：量化组合对诈骗概率的影响
"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 导入机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 导入中文分词 (暂时不需要)
# import jieba

# =========================
# 一、配置部分
# =========================

# 数据文件路径
DATA_PATH = "通话数据互动策略结果\\训练集结果.csv"

# 字段名
TEXT_COLUMN = "specific_dialogue_content"
STRATEGY_COLUMN = "interaction_strategy"
IS_FRAUD_COLUMN = "is_fraud"

# 对话行为类别
SPEECH_ACT_CATEGORIES = ["请求", "陈述", "确认", "拒绝", "其他"]

# =========================
# 二、数据准备函数
# =========================

def load_data(path: str):
    """
    加载对话数据
    """
    try:
        df = pd.read_csv(path, encoding='gbk')
        print(f"数据加载成功，共 {len(df)} 条记录")
        print("字段名：", df.columns.tolist())
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def extract_left_sentences(dialogue_content):
    """
    从对话内容中提取所有left端的句子
    返回句子列表
    """
    if pd.isna(dialogue_content):
        return []

    # 按行分割对话
    lines = str(dialogue_content).split('\n')

    left_sentences = []
    for line in lines:
        line = line.strip()
        if line.startswith('left:'):
            # 提取left端的发言内容
            sentence = line.replace('left:', '').strip()
            if sentence:  # 只保留非空句子
                left_sentences.append(sentence)

    return left_sentences

def preprocess_dialogue_data(df):
    """
    数据预处理：提取left端句子，准备用于行为分析
    """
    print("开始预处理对话数据...")

    # 为每条对话提取left端句子
    df['left_sentences'] = df[TEXT_COLUMN].apply(extract_left_sentences)

    # 过滤掉没有left句子的对话
    df_filtered = df[df['left_sentences'].apply(len) > 0].copy()

    print(f"预处理完成，有效对话数：{len(df_filtered)}")

    return df_filtered

# =========================
# 三、对话行为识别函数
# =========================

def create_speech_act_prompt(sentence):
    """
    创建用于对话行为识别的prompt
    """
    prompt = f"""请判断以下句子的对话行为类别，并只输出类别名称。

对话行为类别包括：
- 请求：询问、索取信息、提出要求
- 陈述：提供信息、描述事实
- 确认：肯定、确认对方观点
- 拒绝：否定、拒绝请求或提议
- 其他：不属于以上类别的发言

句子：{sentence}

请只输出类别名称："""

    return prompt

def mock_llm_predict(sentence):
    """
    模拟LLM预测对话行为类别
    使用规则-based方法进行快速预测
    """
    sentence_lower = sentence.lower()

    if any(word in sentence_lower for word in ['吗', '呢', '什么', '怎么', '能否', '可以']):
        return "请求"
    elif any(word in sentence_lower for word in ['是', '有', '没有', '就是', '但是']):
        return "陈述"
    elif any(word in sentence_lower for word in ['是的', '对的', '好的', '没错']):
        return "确认"
    elif any(word in sentence_lower for word in ['不', '不行', '拒绝', '不同意']):
        return "拒绝"
    else:
        return "其他"

def predict_speech_acts_for_dialogue(sentences):
    """
    为对话中的所有句子预测行为类别
    返回行为类别计数向量
    """
    predictions = []
    for sentence in sentences:
        # 使用模拟预测
        prediction = mock_llm_predict(sentence)
        predictions.append(prediction)

    # 统计各类别数量
    counts = Counter(predictions)

    # 转换为固定顺序的向量
    vector = [counts.get(cat, 0) for cat in SPEECH_ACT_CATEGORIES]

    return vector, predictions

def add_speech_act_features(df):
    """
    为数据框添加对话行为特征
    """
    print("开始为对话添加行为类别特征...")

    # 为每条对话计算行为类别向量
    speech_act_results = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"处理第 {idx+1}/{len(df)} 条对话...")

        vector, predictions = predict_speech_acts_for_dialogue(row['left_sentences'])
        speech_act_results.append({
            'speech_category_vector': vector,
            'speech_category_details': predictions,
            'total_sentences': len(row['left_sentences'])
        })

    # 将结果添加到数据框
    results_df = pd.DataFrame(speech_act_results)
    df_with_features = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    print("行为类别特征添加完成")

    return df_with_features

# =========================
# 四、联合分析函数
# =========================

def analyze_joint_patterns(df):
    """
    分析行为×策略×诈骗的联合模式
    """
    print("开始联合模式分析...")

    results = {}

    # 按is_fraud分组
    for fraud_label in [0, 1]:
        fraud_df = df[df[IS_FRAUD_COLUMN] == fraud_label]
        fraud_name = "诈骗" if fraud_label == 1 else "非诈骗"

        print(f"\n=== {fraud_name}对话分析 ===")

        # 统计策略分布
        strategy_counts = fraud_df[STRATEGY_COLUMN].value_counts()
        print(f"策略分布：{strategy_counts.to_dict()}")

        # 分析行为类别分布
        total_sentences = fraud_df['total_sentences'].sum()
        total_vectors = np.array(fraud_df['speech_category_vector'].tolist())

        # 计算各类别的总占比
        category_totals = total_vectors.sum(axis=0)
        category_percentages = category_totals / total_sentences * 100

        print(f"行为类别分布（占所有句子的百分比）：")
        for cat, pct in zip(SPEECH_ACT_CATEGORIES, category_percentages):
            print(".2f")

        results[fraud_name] = {
            'strategy_dist': strategy_counts,
            'speech_act_dist': dict(zip(SPEECH_ACT_CATEGORIES, category_percentages)),
            'total_dialogues': len(fraud_df),
            'total_sentences': total_sentences
        }

    return results

def analyze_behavior_strategy_combinations(df):
    """
    分析行为×策略组合与诈骗的相关性
    """
    print("\n开始行为×策略组合分析...")

    combinations = []

    for idx, row in df.iterrows():
        strategy = row[STRATEGY_COLUMN]
        speech_vector = row['speech_category_vector']
        is_fraud = row[IS_FRAUD_COLUMN]

        # 为每个行为类别创建组合
        for i, category in enumerate(SPEECH_ACT_CATEGORIES):
            count = speech_vector[i]
            if count > 0:  # 只考虑出现的行为类别
                combinations.append({
                    'strategy': strategy,
                    'speech_act': category,
                    'count': count,
                    'is_fraud': is_fraud
                })

    # 转换为DataFrame便于分析
    combo_df = pd.DataFrame(combinations)

    # 计算每个组合的诈骗概率
    combo_stats = combo_df.groupby(['strategy', 'speech_act']).agg({
        'count': 'sum',
        'is_fraud': ['sum', 'count', 'mean']
    }).round(4)

    combo_stats.columns = ['total_count', 'fraud_count', 'total_combinations', 'fraud_ratio']
    combo_stats = combo_stats.reset_index()

    # 计算相对风险比
    overall_fraud_rate = df[IS_FRAUD_COLUMN].mean()

    combo_stats['risk_ratio'] = combo_stats['fraud_ratio'] / overall_fraud_rate
    # 处理可能的无穷大或NaN值
    combo_stats['risk_ratio'] = combo_stats['risk_ratio'].replace([np.inf, -np.inf], np.nan)
    combo_stats['risk_ratio'] = combo_stats['risk_ratio'].fillna(0)
    combo_stats['risk_ratio'] = combo_stats['risk_ratio'].round(3)

    # 按风险比排序
    combo_stats = combo_stats.sort_values('risk_ratio', ascending=False)

    print("\n行为×策略组合风险分析（前10个高风险组合）：")
    print(combo_stats.head(10)[['strategy', 'speech_act', 'fraud_ratio', 'risk_ratio', 'total_count']].to_string())

    return combo_stats

# =========================
# 五、解释建模函数
# =========================

def build_explainable_model(df):
    """
    构建可解释的诈骗预测模型
    """
    print("\n开始构建解释模型...")

    # 准备特征
    feature_data = []

    for idx, row in df.iterrows():
        features = {}

        # 策略one-hot编码
        strategy = row[STRATEGY_COLUMN]
        features[f'strategy_{strategy}'] = 1

        # 行为类别占比
        speech_vector = row['speech_category_vector']
        total_sentences = row['total_sentences']

        if total_sentences > 0:
            for i, category in enumerate(SPEECH_ACT_CATEGORIES):
                features[f'{category}_ratio'] = speech_vector[i] / total_sentences
        else:
            for category in enumerate(SPEECH_ACT_CATEGORIES):
                features[f'{category}_ratio'] = 0

        # 行为×策略组合特征
        for i, category in enumerate(SPEECH_ACT_CATEGORIES):
            if total_sentences > 0:
                ratio = speech_vector[i] / total_sentences
                features[f'{strategy}_{category}'] = ratio

        feature_data.append(features)

    # 转换为DataFrame
    feature_df = pd.DataFrame(feature_data)
    feature_df[IS_FRAUD_COLUMN] = df[IS_FRAUD_COLUMN].values

    # 填充缺失值
    feature_df = feature_df.fillna(0)

    # 分离特征和标签
    X = feature_df.drop(IS_FRAUD_COLUMN, axis=1)
    y = feature_df[IS_FRAUD_COLUMN]

    # 确保标签是数值类型
    y = y.astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练逻辑回归模型
    print("训练逻辑回归模型...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)

    # 训练决策树模型
    print("训练决策树模型...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)

    # 评估模型
    models = {'LogisticRegression': lr_model, 'DecisionTree': dt_model}

    for name, model in models.items():
        print(f"\n=== {name} 模型性能 ===")
        y_pred = model.predict(X_test)

        print(".4f")
        print("分类报告：")
        print(classification_report(y_test, y_pred, digits=4))

        # 特征重要性分析
        if hasattr(model, 'coef_'):
            # 逻辑回归系数
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.coef_[0]
            })
        else:
            # 决策树特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })

        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False, key=abs)

        print("\n前10个重要特征：")
        print(feature_importance.head(10).to_string())

    return lr_model, dt_model, feature_df

# =========================
# 六、主流程
# =========================

def main():
    print("===== 实验二：对话行为与交互策略建模 =====")

    # 1. 数据准备
    print("\n1. 数据准备阶段")
    df = load_data(DATA_PATH)
    if df is None:
        return

    df_processed = preprocess_dialogue_data(df)
    print(f"处理后的数据形状：{df_processed.shape}")

    # 2. 对话行为识别
    print("\n2. 对话行为识别阶段")
    df_with_features = add_speech_act_features(df_processed)

    # 3. 联合分析
    print("\n3. 联合分析阶段")
    pattern_results = analyze_joint_patterns(df_with_features)
    combo_stats = analyze_behavior_strategy_combinations(df_with_features)

    # 4. 解释建模
    print("\n4. 解释建模阶段")
    lr_model, dt_model, feature_df = build_explainable_model(df_with_features)

    # 保存结果
    print("\n保存分析结果...")
    combo_stats.to_csv('behavior_strategy_analysis.csv', index=False, encoding='utf-8-sig')
    feature_df.to_csv('model_features.csv', index=False, encoding='utf-8-sig')

    print("\n实验完成！结果已保存到CSV文件中。")

if __name__ == "__main__":
    main()
