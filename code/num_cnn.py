import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========================
# 🔧 配置区域
# ========================
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
EPOCHS = 10
MODEL_PATH = "../pth/mnist_best_cnn.pth"

# 自动选择设备
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# ========================
# 📦 数据处理
# ========================
transform = transforms.Compose([
    transforms.RandomRotation(10),  # 数据增强
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="../data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE)

# ========================
# 🧠 模型定义（高性能 CNN）
# ========================
class BestMNISTModel(nn.Module):
    def __init__(self):
        super(BestMNISTModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

model = BestMNISTModel().to(device)

# ========================
# ⚙️ 损失函数与优化器
# ========================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========================
# 🏋️ 训练函数
# ========================
def train(epoch):
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Epoch {epoch} [{batch * len(X)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

# ========================
# 🧪 测试函数
# ========================
def test():
    model.eval()
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item() * y.size(0)
            pred = output.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / total
    print(f"\n✅ 测试准确率: {acc:.2f}%, 测试损失: {avg_loss:.4f}\n")

# ========================
# 🚀 主训练循环
# ========================
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test()

# ========================
# 💾 保存模型
# ========================
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 模型参数已保存至: {MODEL_PATH}")
