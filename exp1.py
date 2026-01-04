# -*- coding: utf-8 -*-
"""
实验一：基于中文对话的诈骗检测（SVM 版本）
在 Windows 上可运行的完整示例代码

功能：
1. 加载对话数据（CSV）
2. 使用 TF-IDF + SVM 进行二分类（非诈骗=0 / 诈骗=1）
3. 计算整体准确率、诈骗样本准确率、非诈骗样本准确率
4. 保存测试集预测结果到 CSV 文件
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import jieba


# =========================
# 一、配置部分（按需要修改）
# =========================

# 数据文件路径（请把你的数据文件名改到这里）
DATA_PATH = "通话数据互动策略结果\训练集结果.csv"  # 比如: "dialogue_dataset.csv"

# 文本和标签的字段名（需与数据集中的列名一致）
TEXT_COLUMN = "specific_dialogue_content"
LABEL_COLUMN = "is_fraud"

# 测试集占比
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =========================
# 二、辅助函数
# =========================

def chinese_tokenizer(text: str):
    """
    使用 jieba 对中文文本进行分词，返回词列表。
    如果文本为空或缺失，返回空列表。
    """
    if pd.isna(text):
        return []
    # cut 返回的是生成器，这里转成 list
    return list(jieba.cut(str(text)))


def load_data(path: str, text_col: str, label_col: str):
    """
    从 CSV 文件中加载数据，并检查基本字段。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}，请确认路径是否正确。")

    df = pd.read_csv(path, encoding="utf-8")

    # 检查字段
    if text_col not in df.columns:
        raise ValueError(f"数据集中缺少文本字段 '{text_col}'，"
                         f"当前字段有：{list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"数据集中缺少标签字段 '{label_col}'，"
                         f"当前字段有：{list(df.columns)}")

    # 只保留需要的两列，并去掉缺失值
    df = df[[text_col, label_col]].dropna()

    # 标签转为 int（0 / 1）
    df[label_col] = df[label_col].astype(int)

    return df


def compute_class_accuracies(y_true, y_pred):
    """
    计算：
    - 整体准确率
    - 诈骗类准确率 accuracy_fraud
    - 非诈骗类准确率 accuracy_nonfraud
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 整体准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 诈骗样本（标签=1）
    fraud_mask = (y_true == 1)
    if fraud_mask.sum() > 0:
        accuracy_fraud = accuracy_score(y_true[fraud_mask], y_pred[fraud_mask])
    else:
        accuracy_fraud = np.nan  # 没有诈骗样本时，无法计算

    # 非诈骗样本（标签=0）
    nonfraud_mask = (y_true == 0)
    if nonfraud_mask.sum() > 0:
        accuracy_nonfraud = accuracy_score(y_true[nonfraud_mask], y_pred[nonfraud_mask])
    else:
        accuracy_nonfraud = np.nan

    return accuracy, accuracy_fraud, accuracy_nonfraud


# =========================
# 三、主流程
# =========================

def main():
    print("===== 实验一：诈骗对话识别（SVM 模型）=====")
    print(f"数据路径: {DATA_PATH}")
    print("开始加载数据...")

    # 1. 加载数据
    df = load_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
    print("数据加载完成！")
    print("数据条数:", len(df))
    print(df.head())

    X = df[TEXT_COLUMN].astype(str).tolist()
    y = df[LABEL_COLUMN].values

    # 2. 划分训练集和测试集
    print("\n开始划分训练集 / 测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # 保证类别比例
    )
    print(f"训练集大小: {len(X_train)}，测试集大小: {len(X_test)}")

    # 3. 文本向量化（TF-IDF）
    print("\n构建 TF-IDF 特征...")

    # 使用 jieba 分词，ngram 可以在 1-2 之间尝试
    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        ngram_range=(1, 2),  # 一元、二元词组
        max_features=5000,   # 特征数量上限，可根据数据大小调整
        min_df=2             # 至少在 2 个样本中出现
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF 特征维度:", X_train_tfidf.shape[1])

    # 4. 训练分类模型（SVM）
    print("\n开始训练 SVM 模型...")
    clf = LinearSVC(random_state=RANDOM_STATE)
    clf.fit(X_train_tfidf, y_train)
    print("模型训练完成！")

    # 5. 在训练集和测试集上进行预测
    print("\n对训练集进行预测...")
    y_train_pred = clf.predict(X_train_tfidf)
    print("对测试集进行预测...")
    y_test_pred = clf.predict(X_test_tfidf)

    # 6. 计算分类性能指标
    print("\n===== 训练集性能 =====")
    train_acc, train_acc_fraud, train_acc_nonfraud = compute_class_accuracies(
        y_train, y_train_pred
    )
    print(f"训练集整体准确率 accuracy_train = {train_acc:.4f}")
    print(f"训练集诈骗类准确率 accuracy_fraud_train = {train_acc_fraud:.4f}")
    print(f"训练集非诈骗类准确率 accuracy_nonfraud_train = {train_acc_nonfraud:.4f}")

    print("\n===== 测试集性能 =====")
    test_acc, test_acc_fraud, test_acc_nonfraud = compute_class_accuracies(
        y_test, y_test_pred
    )
    print(f"测试集整体准确率 accuracy = {test_acc:.4f}")
    print(f"测试集诈骗类准确率 accuracy_fraud = {test_acc_fraud:.4f}")
    print(f"测试集非诈骗类准确率 accuracy_nonfraud = {test_acc_nonfraud:.4f}")

    print("\n===== 测试集分类报告（precision/recall/F1）=====")
    print(classification_report(y_test, y_test_pred, digits=4))

    print("===== 测试集混淆矩阵 =====")
    cm = confusion_matrix(y_test, y_test_pred)
    print("行：真实标签  列：预测标签")
    print("标签顺序：[0=非诈骗, 1=诈骗]")
    print(cm)

    # 7. 保存测试集预测结果到 CSV 文件
    print("\n开始保存测试集预测结果到 test_predictions.csv ...")
    test_results = pd.DataFrame({
        TEXT_COLUMN: X_test,
        "true_label": y_test,
        "pred_label": y_test_pred
    })
    # 是否预测正确
    test_results["correct"] = (test_results["true_label"] == test_results["pred_label"]).astype(int)

    # 保存
    output_path = "test_predictions.csv"
    test_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存到：{output_path}")

    print("\n实验流程完成！你可以根据错分样本（correct=0）进一步分析模型不足。")


if __name__ == "__main__":
    main()
