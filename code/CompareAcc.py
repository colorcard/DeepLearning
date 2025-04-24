import pandas as pd

# ========================
# 配置区域
# ========================
PREDICT_CSV = "submission.csv"       # 你的模型预测结果
ANSWER_CSV = "Digit/digit_test_dataset/labels.csv"  # 官方参考答案

# ========================
# 准确率计算
# ========================
def compare_accuracy(pred_csv, ans_csv):
    pred_df = pd.read_csv(pred_csv)
    ans_df = pd.read_csv(ans_csv)

    # 按 id 排序，确保一一对应
    pred_df = pred_df.sort_values("id").reset_index(drop=True)
    ans_df = ans_df.sort_values("id").reset_index(drop=True)

    # 对比两个 label 列
    correct = (pred_df["label"] == ans_df["label"]).sum()
    total = len(ans_df)
    accuracy = 100. * correct / total

    print(f"✅ 对比完成：Accuracy = {accuracy:.2f}% ({correct}/{total})")

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    compare_accuracy(PREDICT_CSV, ANSWER_CSV)