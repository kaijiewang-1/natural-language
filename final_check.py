# -*- coding: utf-8 -*-
"""
最终检查脚本
验证所有实验代码和依赖是否正常工作
"""

import sys
import os
import subprocess

def check_python_version():
    """检查Python版本"""
    print("🐍 Python版本检查...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 符合要求")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - 需要3.8+")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n📦 依赖包检查...")

    required_packages = [
        'pandas', 'numpy', 'sklearn', 'jieba', 'matplotlib', 'seaborn'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  缺少依赖包，请运行: pip install {' '.join(missing_packages)}")
        return False

    return True

def check_code_syntax():
    """检查代码语法"""
    print("\n🔍 代码语法检查...")

    python_files = [
        'exp1.py',
        'exp2.py',
        'adversarial_rewrite.py',
        '实验结果展示.py'
    ]

    for file in python_files:
        if os.path.exists(file):
            try:
                subprocess.run([sys.executable, '-m', 'py_compile', file],
                             check=True, capture_output=True)
                print(f"✅ {file} - 语法正确")
            except subprocess.CalledProcessError as e:
                print(f"❌ {file} - 语法错误: {e}")
                return False
        else:
            print(f"❌ {file} - 文件不存在")
            return False

    return True

def check_data_files():
    """检查数据文件"""
    print("\n📊 数据文件检查...")

    required_files = [
        '通话数据互动策略结果/训练集结果.csv',
        'behavior_strategy_analysis.csv',
        'attack_results.csv'
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - 存在")
        else:
            print(f"❌ {file} - 不存在")
            return False

    return True

def check_documentation():
    """检查文档文件"""
    print("\n📚 文档文件检查...")

    required_docs = [
        '大作业-对抗性数据改写在欺诈对话检测中的应用.md',
        'README.md',
        'requirements.txt',
        'LICENSE'
    ]

    for doc in required_docs:
        if os.path.exists(doc):
            print(f"✅ {doc} - 存在")
        else:
            print(f"❌ {doc} - 不存在")
            return False

    return True

def run_basic_tests():
    """运行基本功能测试"""
    print("\n🧪 基本功能测试...")

    try:
        # 测试exp1导入
        from exp1 import load_data, compute_class_accuracies
        print("✅ exp1.py - 核心函数导入成功")

        # 测试exp2导入
        from exp2 import mock_llm_predict, predict_speech_acts_for_dialogue
        print("✅ exp2.py - 核心函数导入成功")

        # 测试adversarial_rewrite导入
        from adversarial_rewrite import AdversarialRewriter
        print("✅ adversarial_rewrite.py - 核心类导入成功")

        # 测试基本功能
        test_sentence = "您好，需要验证您的账户信息吗？"
        result = mock_llm_predict(test_sentence)
        print(f"✅ 对话行为预测测试: '{test_sentence}' -> {result}")

        return True

    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False

def main():
    """主检查函数"""
    print("🔍 大作业最终检查")
    print("=" * 50)

    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("代码语法", check_code_syntax),
        ("数据文件", check_data_files),
        ("文档文件", check_documentation),
        ("基本功能", run_basic_tests)
    ]

    results = []
    for name, check_func in checks:
        print(f"\n🔎 检查: {name}")
        result = check_func()
        results.append((name, result))

    # 总结报告
    print("\n" + "=" * 50)
    print("📋 检查结果总结")
    print("=" * 50)

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print("20")
        if result:
            passed += 1

    print(f"\n🎯 总体结果: {passed}/{total} 项检查通过")

    if passed == total:
        print("\n🎉 恭喜！所有检查都通过了，大作业准备就绪！")
        print("\n📝 提交提醒:")
        print("1. 确保GitHub仓库已创建并上传所有代码")
        print("2. 在大作业论文中注明GitHub仓库链接")
        print("3. 检查论文格式是否符合要求")
        print("4. 准备好答辩所需的演示材料")
    else:
        print(f"\n⚠️  有 {total - passed} 项检查未通过，请修复后再提交。")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
