import torch
from csv_to_tensor import VoltageDataset # 导入你的数据工具
from torch.utils.data import DataLoader
from plant_pet_model import MultiModalEncoder     # 导入你的模型类
import torch.nn as nn
import torch.optim as optim

SEQ_LEN = 100
BATCH_SIZE = 8
IMP_DIM = 50  # 假设阻抗维度
HIDDEN = 64
OUT = 3
LR = 0.001

# --- 2. 数据准备 (只需在开始时运行) ---
# 加载时序电压数据
dataset = VoltageDataset('26-03-07 15_30_33_925.csv', seq_len=SEQ_LEN)
# 使用 DataLoader 自动处理 Batch 分组和洗牌（Shuffle）
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()

model = MultiModalEncoder(1, IMP_DIM, HIDDEN, OUT)

optimizer = optim.Adam(model.parameters(), lr=LR)



print("\n开始训练反向传播逻辑...")

# --- 4. 模拟训练循环 ---
for epoch in range(10):
    total_loss = 0.0
    for i, x_time in enumerate(train_loader):
        # 准备数据
        x_imp = torch.randn(x_time.size(0), 50) # 模拟阻抗输入
        labels = torch.randint(0, 3, (x_time.size(0),)) # 模拟标签 [Batch_size]

        # --- 核心训练步 (The 5 Steps) ---
        
        # 1. 梯度归零 (PyTorch 默认累加梯度，每步必须手动清空)
        optimizer.zero_grad()
        
        # 2. 前向传播
        output, _ = model(x_time, x_imp)
        
        # 3. 计算 Loss
        loss = criterion(output, labels)
        
        # 4. 反向传播 (计算计算图中所有参数的梯度)
        loss.backward()
        
        # 5. 权重更新
        optimizer.step()
        
        total_loss += loss.item()

    # 打印每个 Epoch 的平均损失
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/10], Loss: {avg_loss:.4f}")

print("\n训练跑通！")