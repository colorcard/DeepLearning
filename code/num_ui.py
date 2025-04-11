import torch
from torch import nn
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# ========================
# 🔧 配置区域
# ========================
MODEL_PATH = "../pth/mnist_best_cnn.pth"  # 要与训练脚本中保存的路径一致
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备：{device}")

# ========================
# 🧠 CNN 模型结构
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

# ========================
# 📦 加载模型函数
# ========================
def build_model():
    return CNNModel()

def load_model(path=MODEL_PATH):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ========================
# 🎨 手写界面部分
# ========================
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✍️ 手写数字识别 (CNN版)")

        self.canvas_size = 280
        self.image_size = 28

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="🧹 清除", command=self.clear_canvas).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="🔍 识别", command=self.predict_digit).pack(side=tk.LEFT, padx=10)

        self.label = tk.Label(root, text="", font=("Helvetica", 20))
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        r = 10
        x1, y1 = event.x - r, event.y - r
        x2, y2 = event.x + r, event.y + r
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.label.config(text="")

    def predict_digit(self):
        # 图像预处理
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(1).item()

        self.label.config(text=f"预测结果是： {pred}")

# ========================
# 🚀 主程序入口
# ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()