import torch
import torch.nn as nn
import time


def benchmark_matrix_multiplication(device, num_iterations=100):
    # 测试大矩阵乘法性能，矩阵尺寸为2048x2048
    size = (2048, 2048)
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    # 暖身运行，避免初始化开销干扰测量
    for _ in range(10):
        _ = torch.mm(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = torch.mm(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(
        f"[矩阵乘法] 设备: {device}, {num_iterations}次迭代总耗时: {elapsed:.4f}秒, 平均单次: {elapsed / num_iterations:.4f}秒")


def benchmark_convolution(device, num_iterations=100):
    # 测试卷积操作性能，假设输入图片尺寸为224x224，Batch大小为32
    batch_size = 32
    in_channels = 3
    out_channels = 64
    height = width = 224
    kernel_size = 3
    # 构造随机输入图片和卷积层
    input_tensor = torch.randn(batch_size, in_channels, height, width, device=device)
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1).to(device)
    # 暖身运行
    for _ in range(10):
        _ = conv_layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = conv_layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(
        f"[卷积操作] 设备: {device}, {num_iterations}次迭代总耗时: {elapsed:.4f}秒, 平均单次: {elapsed / num_iterations:.4f}秒")


def benchmark_nn_forward(device, num_iterations=100):
    # 测试多层全连接网络前向传播性能
    input_size = 2048
    hidden_size = 1024
    output_size = 10
    batch_size = 64
    # 定义一个具有两层隐藏层的全连接网络
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    ).to(device)
    # 构造随机输入
    input_tensor = torch.randn(batch_size, input_size, device=device)
    # 暖身运行
    for _ in range(10):
        _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(
        f"[神经网络前向传播] 设备: {device}, {num_iterations}次迭代总耗时: {elapsed:.4f}秒, 平均单次: {elapsed / num_iterations:.4f}秒")


def main():
    # 检测可用设备：CPU、CUDA和MPS
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for device in devices:
        print(f"\n=== 测试设备: {device} ===")
        benchmark_matrix_multiplication(device)
        benchmark_convolution(device)
        benchmark_nn_forward(device)


if __name__ == "__main__":
    main()