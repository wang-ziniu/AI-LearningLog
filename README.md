# AI-LearningLog

## Notes1
1. **张量核心操作**
   - GPU张量创建与矩阵运算
   - 自动微分机制（`requires_grad`与`backward()`）
2. **数据处理**
   - `Dataset`与`DataLoader`构建
   - 自定义猫狗数据集实现
   - MNIST标准化预处理
3. **数据可视化**
   - Matplotlib单图/多图显示
   - 数据标签验证检查
4. **经典模型实现**
   - LeNet-5架构完整实现
   - 卷积层、池化层、全连接层组合
5. **完整训练流程**
   - CUDA设备迁移（模型&数据）
   - 训练循环构建（前向/反向传播）
   - 优化器（SGD with momentum）配置

#### 关键技术组件
- `torchvision.transforms` 数据增强
- `nn.Module` 模型基类继承
- 交叉熵损失函数（CrossEntropyLoss）
- 梯度清零机制（optimizer.zero_grad()）


相关代码详见：[`Notes1.py`](Notes1.py) 
