import json
import os
import numpy as np
from scipy import stats

# 当前目录，假设脚本和 .jsonl 文件放在同一文件夹
data_dir = "/home/chenchen/gjx/Judge/data/ours/bias/gpt5"

# 根据你截图中的文件名提取的 bias 类别
bias_types = [
    "rich_content", "chain_of_thought", "verbosity", "reference", 
    "gender", "factual_error", "compassion-fade", "bandwagon", 
    "distraction", "sentiment", "diversity"
]

def load_scores(filepath):
    """从 .jsonl 文件中提取 score 并返回 numpy 数组"""
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # 跳过空行
                data = json.loads(line)
                scores.append(float(data['score']))
    return np.array(scores)

def main():
    # 1. 加载 Clean 组（控制组）的基准数据
    clean_file = os.path.join(data_dir, "clean_50p_gpt4o.jsonl")
    try:
        clean_scores = load_scores(clean_file)
        clean_mean = np.mean(clean_scores)
        clean_std = np.std(clean_scores, ddof=1) # ddof=1 表示样本标准差
        print(f"=== 基准数据 (Clean) ===")
        print(f"样本量: {len(clean_scores)}")
        print(f"Clean Mean ± Std: {clean_mean:.2f} ± {clean_std:.2f}\n")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {clean_file}，请检查路径。")
        return

    # 2. 遍历所有的 Bias 组计算统计数据并进行显著性检验
    print(f"{'Bias Type':<18} | {'Mean ± Std':<15} | {'P-Value':<10} | {'显著性'}")
    print("-" * 65)

    for bias in bias_types:
        bias_file = os.path.join(data_dir, f"{bias}_50p_gpt4o.jsonl")
        
        if not os.path.exists(bias_file):
            print(f"{bias:<18} | 文件未找到")
            continue
            
        bias_scores = load_scores(bias_file)
        
        # 确保数据对齐（必须都是 50 个样本才能做配对 t 检验）
        if len(bias_scores) != len(clean_scores):
            print(f"{bias:<18} | 错误: 样本量不匹配 (Clean:{len(clean_scores)}, Bias:{len(bias_scores)})")
            continue

        # 计算均值和标准差
        mean_val = np.mean(bias_scores)
        std_val = np.std(bias_scores, ddof=1)
        
        # 执行配对样本 t 检验
        # 检验 "clean_scores" 和 "bias_scores" 是否存在显著差异
        t_stat, p_val = stats.ttest_rel(clean_scores, bias_scores)
        
        # 标记显著性星号，方便你直接写进论文表格
        if p_val < 0.001:
            sig = "*** (p<0.001)"
        elif p_val < 0.01:
            sig = "** (p<0.01)"
        elif p_val < 0.05:
            sig = "* (p<0.05)"
        else:
            sig = "ns (不显著)"

        # 打印格式化输出
        print(f"{bias:<18} | {mean_val:.2f} ± {std_val:.2f}   | {p_val:.4f}   | {sig}")

if __name__ == "__main__":
    main()