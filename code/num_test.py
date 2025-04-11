# 导入 PyTorch 和相关模块
import torch
from torch import nn                            # 神经网络模块
from torchvision import datasets, transforms    # 常用数据集和预处理
from torch.utils.data import DataLoader         # 批量数据加载器

# ========================
# 🧩 配置区域（可快速调整）
# ========================
BATCH_SIZE = 64             # 每批训练数据的大小
TEST_BATCH_SIZE = 1000      # 每批测试数据的大小
HIDDEN_SIZE = 128           # 隐藏层神经元数量（MLP 中间层维度）
LEARNING_RATE = 1e-3        # 学习率（控制模型更新速度）
EPOCHS = 5                  # 总训练轮数
DATA_PATH = "../data"       # MNIST 数据保存路径
SAVE = True             # 是否保存模型参数

# 自动选择计算设备：优先使用 M1/M2/M3 芯片的 MPS 加速（Mac），否则使用 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")  # 打印使用的设备类型

# 定义数据预处理方式：将图像转换为张量（Tensor），值缩放到 0~1
transform = transforms.ToTensor()

# 加载训练数据集（MNIST 手写数字图片）
train_data = datasets.MNIST(
    root=DATA_PATH,        # 下载路径
    train=True,            # 训练集
    download=True,         # 如果没有就自动下载
    transform=transform    # 应用预处理
)

# 加载测试数据集
test_data = datasets.MNIST(
    root=DATA_PATH,
    train=False,           # 测试集
    download=True,
    transform=transform
)

# 构造训练数据加载器，支持批量读取并打乱顺序
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# 构造测试数据加载器，不打乱顺序
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE)

# 构建一个简单的全连接神经网络（MLP）
model = nn.Sequential(
    nn.Flatten(),                      # 将 28x28 图像展平为 784 向量
    nn.Linear(28 * 28, HIDDEN_SIZE),   # 第一个全连接层：输入784维，输出隐藏层维度
    nn.ReLU(),                         # 激活函数：引入非线性
    nn.Linear(HIDDEN_SIZE, 10)         # 第二个全连接层：输出10类（数字0~9）
).to(device)                           # 把模型放到指定设备上（MPS 或 CPU）

# 定义损失函数：交叉熵（分类问题的常用选择）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器：Adam（自动调节学习率的优化算法）
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 定义训练过程（执行一次完整 epoch）
def train(epoch):
    model.train()  # 设置模型为训练模式
    for batch, (X, y) in enumerate(train_loader):  # 遍历每一批数据
        X, y = X.to(device), y.to(device)  # 将数据移动到计算设备

        pred = model(X)             # 前向传播：获得预测结果
        loss = loss_fn(pred, y)     # 计算损失值（预测 vs 真值）

        optimizer.zero_grad()       # 清除上一步残留的梯度
        loss.backward()             # 反向传播：计算梯度
        optimizer.step()            # 执行参数更新（优化）

        if batch % 100 == 0:  # 每隔100个 batch 打印一次当前损失
            print(f"Epoch {epoch} [{batch * len(X)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

# 定义测试过程（不会更新模型参数）
def test():
    model.eval()  # 设置为评估模式（关闭 Dropout 等训练特性）
    correct = 0                # 正确预测数
    total = 0                  # 总样本数
    total_loss = 0             # 累计损失
    with torch.no_grad():     # 禁用梯度计算（节省资源）
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)                           # 前向传播，预测结果
            loss = loss_fn(output, y)                   # 计算当前批次的损失
            total_loss += loss.item() * y.size(0)       # 乘上样本数量，加权累积
            pred = output.argmax(1)                     # 获取每个样本预测的类别（最大值位置）
            correct += (pred == y).sum().item()         # 累加预测正确的数量
            total += y.size(0)                          # 累加样本数量

    avg_loss = total_loss / total     # 计算平均损失
    acc = 100. * correct / total      # 计算准确率（百分比）
    print(f"Test Accuracy: {acc:.2f}%, Test Loss: {avg_loss:.4f}\n")

# 训练主循环：重复执行训练 + 测试
for epoch in range(1, EPOCHS + 1):
    train(epoch)  # 训练一轮
    test()        # 测试一轮

# 保存模型参数到文件（如：mnist_mlp.pth）
if SAVE:
    torch.save(model.state_dict(), "../pth/mnist_mlp.pth")
    print("✅ 模型已保存为 mnist_mlp.pth")