# AI 学习笔记：从张量操作到深度学习模型训练

# --- 1. 张量（Tensor）操作 ---
# 张量是 PyTorch 中的核心数据结构，可以看作是多维数组。

import torch

# 创建张量
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cuda')  # 在 GPU 上创建张量
y = torch.randn(2, 2).cuda()  # 创建随机张量并转移到 GPU

# 矩阵乘法
z = torch.mm(x, y)  # 矩阵乘法

# 自动求导
w = torch.randn(2, 2, requires_grad=True, device='cuda')  # 需要计算梯度
loss = z.sum()  # 定义损失函数
loss.backward()  # 反向传播计算梯度
print(w.grad)  # 打印梯度

# --- 2. 数据加载与预处理 ---
# 在深度学习中，数据加载和预处理是非常重要的步骤。

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的归一化参数
])

# 加载 MNIST 数据集
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)

# 数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)

# 自定义数据集
import os
from PIL import Image

class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 0 if 'cat' in img_name else 1  # 猫为 0，狗为 1

        if self.transform:
            image = self.transform(image)
        return image, label

# --- 3. 数据可视化 ---
# 在训练模型之前，可以通过可视化检查数据是否正确加载。

import matplotlib.pyplot as plt

# 可视化单张图片
for images, labels in train_loader:
    plt.imshow(images[0].squeeze(), cmap="gray")  # 显示第一张图片
    plt.title(f"Label: {labels[0].item()}")  # 显示标签
    plt.show()
    break

# 可视化多个图片
fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 创建 1 行 5 列的子图
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap="gray")
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis("off")
plt.show()

# --- 4. 深度学习模型：LeNet-5 ---
# LeNet-5 是一个经典的卷积神经网络，用于图像分类任务。

import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道 1，输出通道 6，卷积核 5x5
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，窗口大小 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道 6，输出通道 16，卷积核 5x5
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 全连接层，输入维度 16*4*4，输出维度 120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入维度 120，输出维度 84
        self.fc3 = nn.Linear(84, 10)  # 输出层，10 个类别

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # 卷积 + ReLU + 池化
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 卷积 + ReLU + 池化
        x = x.view(-1, 16 * 4 * 4)  # 展平
        x = nn.functional.relu(self.fc1(x))  # 全连接 + ReLU
        x = nn.functional.relu(self.fc2(x))  # 全连接 + ReLU
        x = self.fc3(x)  # 输出层
        return x

# --- 5. 模型训练 ---
# 训练流程包括定义模型、损失函数、优化器，以及训练循环。

# 定义模型
model = LeNet5().cuda()  # 将模型加载到 GPU

# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降

# 训练循环
for epoch in range(10):  # 训练 10 个 epoch
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()  # 将数据加载到 GPU
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 batch 打印一次损失
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0

# --- 6. 总结 ---
# 学到的知识点：
# 1. 张量操作：张量的创建、运算和自动求导。
# 2. 数据加载与预处理：使用内置数据集、自定义数据集类、DataLoader 批量加载数据。
# 3. 数据可视化：使用 matplotlib 显示图片和标签。
# 4. 深度学习模型：定义卷积神经网络（LeNet-5），理解卷积层、池化层和全连接层的作用。
# 5. 模型训练：定义损失函数和优化器，训练循环包括前向传播、反向传播和参数更新。

# 下一步学习方向：
# 1. 模型评估：如何在测试集上评估模型性能，使用准确率、混淆矩阵等指标。
# 2. 超参数调优：调整学习率、批量大小等超参数。
# 3. 迁移学习：使用预训练模型解决复杂任务。
# 4. 更多深度学习模型：学习 ResNet、VGG 等更复杂的网络结构。