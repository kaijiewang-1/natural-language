# -*- coding: utf-8 -*-
"""
对抗性数据改写模块
实现针对欺诈对话的对抗样本生成

功能：
1. 同义词替换攻击
2. 句子重构攻击
3. 模糊化表达攻击
4. 攻击效果评估
"""

import os
import re
import pandas as pd
import numpy as np
import jieba
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 导入机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 导入中文NLP库
# import synonyms  # 需要许可证，暂时不用

# =========================
# 一、配置部分
# =========================

# 数据文件路径
ORIGINAL_DATA_PATH = "通话数据互动策略结果\\训练集结果.csv"
PROCESSED_DATA_PATH = "processed_dialogue_data.csv"  # 实验二处理后的数据

# 输出文件路径
ADVERSARIAL_DATA_PATH = "adversarial_dialogue_data.csv"
ATTACK_RESULTS_PATH = "attack_results.csv"

# =========================
# 二、对抗改写策略类
# =========================

class AdversarialRewriter:
    """
    对抗性改写器
    实现多种对抗样本生成策略
    """

    def __init__(self):
        # 初始化同义词库
        self.synonym_dict = self._build_synonym_dict()

        # 诈骗相关关键词
        self.fraud_keywords = {
            '银行': ['银行', '金融机构', '信用社', '农商行'],
            '账户': ['账户', '账号', '户头', '卡号'],
            '密码': ['密码', '口令', '验证码', 'PIN码'],
            '投资': ['投资', '理财', '基金', '股票'],
            '收益': ['收益', '回报', '利润', '利息'],
            '安全': ['安全', '保护', '保障', '放心'],
            '验证': ['验证', '确认', '核实', '检查'],
            '转账': ['转账', '汇款', '付款', '支付']
        }

    def _build_synonym_dict(self):
        """构建同义词词典"""
        synonym_dict = {}

        # 手动构建常用同义词
        synonym_dict.update({
            '银行': ['银行', '金融机构', '信用社', '农商行', '建设银行', '工商银行'],
            '账户': ['账户', '账号', '户头', '卡号', '账户信息'],
            '密码': ['密码', '口令', '验证码', 'PIN码', '支付密码'],
            '投资': ['投资', '理财', '基金', '股票', '期货'],
            '收益': ['收益', '回报', '利润', '利息', '分红'],
            '安全': ['安全', '保护', '保障', '放心', '可靠'],
            '验证': ['验证', '确认', '核实', '检查', '审核'],
            '转账': ['转账', '汇款', '付款', '支付', '转款'],
            '客服': ['客服', '工作人员', '员工', '服务人员'],
            '经理': ['经理', '主任', '主管', '负责人'],
            '帮助': ['帮助', '协助', '支持', '援助'],
            '问题': ['问题', '疑问', '困难', '麻烦'],
            '解决': ['解决', '处理', '办理', '搞定'],
            '保证': ['保证', '确保', '肯定', '一定'],
            '放心': ['放心', '安心', '放心好了', '别担心']
        })

        return synonym_dict

    def synonym_replacement(self, text, replace_ratio=0.3):
        """
        同义词替换策略
        """
        if pd.isna(text):
            return text

        # 分词
        words = list(jieba.cut(text))

        # 找出可替换的词
        replaceable_words = []
        for i, word in enumerate(words):
            if word in self.synonym_dict and len(self.synonym_dict[word]) > 1:
                replaceable_words.append((i, word))

        if not replaceable_words:
            return text

        # 随机选择要替换的词
        num_to_replace = max(1, int(len(replaceable_words) * replace_ratio))
        selected_indices = np.random.choice(len(replaceable_words),
                                          size=num_to_replace, replace=False)

        # 执行替换
        result_words = words.copy()
        for idx in selected_indices:
            pos, original_word = replaceable_words[idx]
            synonyms_list = self.synonym_dict[original_word]
            # 选择不同于原词的同义词
            available_synonyms = [s for s in synonyms_list if s != original_word]
            if available_synonyms:
                new_word = np.random.choice(available_synonyms)
                result_words[pos] = new_word

        return ''.join(result_words)

    def sentence_restructure(self, text):
        """
        句子重构策略
        改变句子结构但保持语义
        """
        if pd.isna(text):
            return text

        # 简单的句子重构规则
        restructure_patterns = [
            # 疑问句重构
            (r'(.+)(吗|呢|什么|怎么|能否|可以)', r'我想知道\1\2'),
            (r'(.+)(吗|呢)', r'请告诉我\1\2'),

            # 陈述句重构
            (r'^(.+?)，(.+)$', r'\2，\1'),
            (r'^(.+?)因为(.+)$', r'\2，所以\1'),

            # 请求句重构
            (r'^(.+?)需要(.+)$', r'\2是\1需要的'),
            (r'^(.+?)请(.+)$', r'麻烦\1\2'),
        ]

        result = text
        for pattern, replacement in restructure_patterns:
            result = re.sub(pattern, replacement, result)
            if result != text:  # 如果有替换，停止
                break

        return result

    def obfuscation_attack(self, text):
        """
        模糊化表达攻击
        使用更委婉或模糊的表达
        """
        if pd.isna(text):
            return text

        obfuscation_dict = {
            '密码': ['私密信息', '安全码', '验证信息'],
            '转账': ['资金转移', '钱款调动', '账户操作'],
            '投资': ['资金配置', '财富管理', '资产规划'],
            '高收益': ['可观的回报', '不错的收益', '理想的收益'],
            '安全': ['有保障的', '可靠的', '值得信赖的'],
            '验证': ['核实一下', '确认一下', '检查一下'],
            '保证': ['确保', '肯定', '一定'],
            '马上': ['立刻', '立即', '很快'],
            '现在': ['目前', '当下', '眼下']
        }

        result = text
        for key, alternatives in obfuscation_dict.items():
            if key in result:
                replacement = np.random.choice(alternatives)
                result = result.replace(key, replacement, 1)  # 只替换一次

        return result

    def apply_attack(self, text, attack_type='synonym'):
        """
        应用指定的攻击类型
        """
        if attack_type == 'synonym':
            return self.synonym_replacement(text)
        elif attack_type == 'restructure':
            return self.sentence_restructure(text)
        elif attack_type == 'obfuscation':
            return self.obfuscation_attack(text)
        elif attack_type == 'combined':
            # 组合攻击：先模糊化，再重构，最后同义词替换
            text = self.obfuscation_attack(text)
            text = self.sentence_restructure(text)
            text = self.synonym_replacement(text)
            return text
        else:
            return text

# =========================
# 三、对抗攻击评估类
# =========================

class AttackEvaluator:
    """
    对抗攻击效果评估器
    """

    def __init__(self):
        self.models = {}
        self.vectorizer = None

    def load_models(self):
        """
        加载预训练模型
        """
        # 这里应该加载实验一和实验二训练的模型
        # 暂时使用简化版本
        pass

    def train_baseline_models(self, X_train, y_train):
        """
        训练基准模型
        """
        # TF-IDF向量化
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: list(jieba.cut(x)),
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # 训练模型
        self.models['LinearSVC'] = LinearSVC(random_state=42)
        self.models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['DecisionTree'] = DecisionTreeClassifier(random_state=42, max_depth=5)

        for name, model in self.models.items():
            model.fit(X_train_tfidf, y_train)
            print(f"{name} 模型训练完成")

    def evaluate_attack(self, original_texts, adversarial_texts, original_labels):
        """
        评估对抗攻击效果
        """
        results = {}

        # 向量化文本
        original_tfidf = self.vectorizer.transform(original_texts)
        adversarial_tfidf = self.vectorizer.transform(adversarial_texts)

        for model_name, model in self.models.items():
            # 原始数据预测
            original_preds = model.predict(original_tfidf)
            original_acc = accuracy_score(original_labels, original_preds)

            # 改写数据预测
            adversarial_preds = model.predict(adversarial_tfidf)
            adversarial_acc = accuracy_score(original_labels, adversarial_preds)

            # 计算攻击指标
            attack_success_rate = np.mean(original_preds != adversarial_preds)
            accuracy_drop = original_acc - adversarial_acc

            results[model_name] = {
                'original_accuracy': original_acc,
                'adversarial_accuracy': adversarial_acc,
                'accuracy_drop': accuracy_drop,
                'attack_success_rate': attack_success_rate
            }

            print(f"\n{model_name} 攻击效果:")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")

        return results

# =========================
# 四、主流程函数
# =========================

def load_and_preprocess_data(data_path):
    """
    加载并预处理数据
    """
    print("加载数据...")
    try:
        df = pd.read_csv(data_path, encoding='gbk')
        print(f"数据加载成功，共 {len(df)} 条记录")

        # 提取left端句子（复用实验二的逻辑）
        df['left_sentences'] = df['specific_dialogue_content'].apply(
            lambda x: extract_left_sentences(x) if pd.notna(x) else []
        )

        # 过滤有效数据
        df = df[df['left_sentences'].apply(len) > 0].copy()

        # 合并left端句子为单个文本用于分类
        df['text_for_classification'] = df['left_sentences'].apply(
            lambda sentences: ' '.join(sentences)
        )

        # 数据清理：去除NaN值
        df = df.dropna(subset=['text_for_classification', 'is_fraud']).copy()
        df['is_fraud'] = df['is_fraud'].astype(int)

        print(f"预处理完成，有效样本数：{len(df)}")

        return df

    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def extract_left_sentences(dialogue_content):
    """
    从对话内容中提取left端句子
    """
    if pd.isna(dialogue_content):
        return []

    lines = str(dialogue_content).split('\n')
    left_sentences = []

    for line in lines:
        line = line.strip()
        if line.startswith('left:'):
            sentence = line.replace('left:', '').strip()
            if sentence:
                left_sentences.append(sentence)

    return left_sentences

def generate_adversarial_dataset(df, attack_types=['synonym', 'restructure', 'obfuscation', 'combined']):
    """
    生成对抗数据集
    """
    rewriter = AdversarialRewriter()

    adversarial_datasets = {}

    for attack_type in attack_types:
        print(f"\n生成 {attack_type} 攻击数据集...")

        df_attack = df.copy()

        # 对每条对话应用攻击
        adversarial_texts = []
        for idx, row in df_attack.iterrows():
            if idx % 500 == 0:
                print(f"处理第 {idx+1}/{len(df_attack)} 条对话...")

            original_sentences = row['left_sentences']
            attacked_sentences = []

            for sentence in original_sentences:
                attacked_sentence = rewriter.apply_attack(sentence, attack_type)
                attacked_sentences.append(attacked_sentence)

            # 合并为分类文本
            adversarial_text = ' '.join(attacked_sentences)
            adversarial_texts.append(adversarial_text)

        df_attack['adversarial_text'] = adversarial_texts
        adversarial_datasets[attack_type] = df_attack

        print(f"{attack_type} 攻击数据集生成完成")

    return adversarial_datasets

def run_attack_evaluation(df_original, adversarial_datasets):
    """
    运行攻击效果评估
    """
    print("\n开始攻击效果评估...")

    # 准备训练数据（使用原始数据训练模型）
    X_train, X_test, y_train, y_test = train_test_split(
        df_original['text_for_classification'].tolist(),
        df_original['is_fraud'].values,
        test_size=0.2,
        random_state=42,
        stratify=df_original['is_fraud']
    )

    # 初始化评估器
    evaluator = AttackEvaluator()
    evaluator.train_baseline_models(X_train, y_train)

    # 评估每种攻击类型
    all_results = {}

    for attack_type, df_attack in adversarial_datasets.items():
        print(f"\n评估 {attack_type} 攻击效果...")

        # 使用相同的测试集样本
        test_indices = X_test  # 这里需要根据实际数据调整
        test_labels = y_test

        # 获取对应的改写文本
        test_adversarial_texts = []
        for i, test_text in enumerate(X_test):
            # 简化处理：直接使用第一条匹配的记录
            matching_rows = df_attack[df_attack['text_for_classification'] == test_text]
            if len(matching_rows) > 0:
                test_adversarial_texts.append(matching_rows.iloc[0]['adversarial_text'])
            else:
                test_adversarial_texts.append(test_text)  # 如果找不到，使用原文

        # 评估攻击效果
        results = evaluator.evaluate_attack(
            X_test, test_adversarial_texts, y_test
        )

        all_results[attack_type] = results

    return all_results

def save_results(adversarial_datasets, attack_results):
    """
    保存结果
    """
    print("\n保存结果...")

    # 保存改写后的数据集
    for attack_type, df in adversarial_datasets.items():
        output_path = f"adversarial_data_{attack_type}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"保存 {attack_type} 数据集到 {output_path}")

    # 保存攻击结果
    results_summary = []
    for attack_type, model_results in attack_results.items():
        for model_name, metrics in model_results.items():
            results_summary.append({
                'attack_type': attack_type,
                'model': model_name,
                'original_accuracy': metrics['original_accuracy'],
                'adversarial_accuracy': metrics['adversarial_accuracy'],
                'accuracy_drop': metrics['accuracy_drop'],
                'attack_success_rate': metrics['attack_success_rate']
            })

    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(ATTACK_RESULTS_PATH, index=False, encoding='utf-8-sig')
    print(f"保存攻击结果到 {ATTACK_RESULTS_PATH}")

def main():
    """
    主流程
    """
    print("===== 对抗性数据改写实验 =====")

    # 1. 加载和预处理数据
    df = load_and_preprocess_data(ORIGINAL_DATA_PATH)
    if df is None:
        return

    # 2. 生成对抗数据集
    attack_types = ['synonym', 'restructure', 'obfuscation', 'combined']
    adversarial_datasets = generate_adversarial_dataset(df, attack_types)

    # 3. 评估攻击效果
    attack_results = run_attack_evaluation(df, adversarial_datasets)

    # 4. 保存结果
    save_results(adversarial_datasets, attack_results)

    print("\n对抗性改写实验完成！")

if __name__ == "__main__":
    main()
