import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========================
# 🔧 配置区域（可快速修改）
# ========================
HIDDEN_SIZE = 128                          # 隐藏层大小（必须与训练时一致）
MODEL_PATH = "../pth/mnist_best_cnn.pth"        # 模型权重路径
BATCH_SIZE = 1000                          # 测试批次大小
DATA_PATH = "../data"                      # 测试数据存放路径

# 自动选择设备（支持 Mac 的 MPS）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ========================
# 🧠 构建模型（结构需保持一致）
# ========================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
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

def build_model():
    return CNNModel()

# 加载模型权重
def load_model(path=MODEL_PATH):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ========================
# 🧪 评估函数
# ========================
def evaluate(model):
    # 加载测试集
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 使用交叉熵作为评估指标
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item() * y.size(0)  # 加权求和
            pred = output.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    print(f"✅ 评估完成：Accuracy = {accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")

# ========================
# 🚀 主程序入口
# ========================
if __name__ == "__main__":
    model = load_model()
    evaluate(model)