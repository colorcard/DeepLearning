import pandas as pd
import matplotlib.pyplot as plt

# 读取 pickle 文件
mp_file = "LSWMD.pkl"
df = pd.read_pickle(mp_file)

# 获取前100个 waferMap 数据
wafer_maps = df["waferMap"].head(1)

# 设置图片展示参数
plt.figure(figsize=(20, 20))
cols = 1  # 每行展示10张图片
rows = 1  # 总共10行（100张图片）

for i, img in enumerate(wafer_maps):
    plt.subplot(rows, cols, i+1)
    plt.imshow(img)  # 默认显示彩色图片
    plt.axis("off")

plt.tight_layout()
plt.show()