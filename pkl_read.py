import pandas as pd
import matplotlib.pyplot as plt

# 读取 pickle 文件
mp_file = "LSWMD.pkl"
df = pd.read_pickle(mp_file)

# # 1. 查看所有列名
# print("所有列名：")
# print(df.columns.tolist())
#
# # 2. 查看每列的数据类型和空值情况
# print("\nDataFrame 信息：")
# print(df.info())
#
# # 3. 查看前几行，快速感受每列的数据样子
# print("\n前 5 行示例：")
# print(df.head())
#
# # 4. 对数值型列给出一些统计指标（count, mean, std, min, max…）
# print("\n数值型列统计描述：")
# print(df.describe())
#
# # 样本总数
# print("样本总数：", len(df))
# # waferMap 的尺寸
# shapes = df['waferMap'].apply(lambda x: (len(x), len(x[0]))).unique()
# print("waferMap 矩阵尺寸：", shapes)
# # 批次数
# print("批次数量（lotName 唯一值）：", df['lotName'].nunique())
# # waferIndex 可选范围
# print("waferIndex 取值：", sorted(df['waferIndex'].unique()))
# # train/test 分布
# print("trianTestLabel 分布：\n", df['trianTestLabel'].value_counts())
# # failureType 分布
# print("failureType 分布：\n", df['failureType'].value_counts())

# 获取前100个 waferMap 数据
wafer_maps = df["waferMap"].head(100)

# 设置图片展示参数
plt.figure(figsize=(20, 20))
cols = 10  # 每行展示10张图片
rows = 10  # 总共10行（100张图片）

for i, img in enumerate(wafer_maps):
    plt.subplot(rows, cols, i+1)
    plt.imshow(img)  # 默认显示彩色图片
    plt.axis("off")

plt.tight_layout()
plt.show()