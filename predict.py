import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from moe import MoE

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'D:\MoE\mixture-of-experts\data\tukeynew2_cleaned.csv')
# 选择多个特征列以及目标列
features = data[['IN_COD','IN_TP','IN_TN','IN_pH',
			'N-O-E_ORP','N-O-E_DO','N-O_NO',
			'S-O-E_ORP','S-O-E_DO','S-O_NO','EF_COD']]  # 添加或修改为你的特征列名
# features = data[['IN_COD','IN_TP','IN_TN','IN_pH',
# 			'N-O-E_ORP','N-O-E_DO','N-O-E_MLSS',
# 			'S-O-E_ORP','S-O-E_DO','S-O-E_MLSS']]
target = data["EF_COD"].values  # 目标列

# 标准化特征
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# 标准化目标
target_scaler = MinMaxScaler()
target_normalized = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()


# 创建输入序列和标签
N = 24  # 时间窗口大小

X = []
y = []
for i in range(N, len(data)):
    X.append(features_normalized[i-N:i])
    y.append(target_normalized[i])

X = np.array(X)
y = np.array(y)
print(X.shape)
# 假设您已经有了 X 和 y
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=None)  # 保留20%的数据作为验证集

import torch
from torch.utils.data import TensorDataset, DataLoader

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# 创建数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=144, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=144, shuffle=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # 重塑输入以匹配模型期望的形状
    # inputs = inputs.view(inputs.shape[0], -1)
print(inputs.shape)  # 应该输出 (batch_size, 264)

net = MoE(input_size=11, output_size=1, num_experts=1, hidden_size=32, noisy_gating=False, seq_len=24, k=1)
net = net.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


net.train()
for epoch in range(10):  # 迭代10次
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, aux_loss = net(inputs)
        outputs = outputs[:, -1, 0].unsqueeze(1)  # 调整输出以匹配标签形状
        loss = criterion(outputs, labels)
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 在验证集上评估模型
    val_loss = 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs, _ = net(inputs)
            outputs = outputs[:, -1, 0].unsqueeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')


print('Finished Training')



from sklearn.metrics import mean_squared_error

net.eval()  # 切换到评估模式
predictions = []
targets = []


with torch.no_grad():

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs, _ = net(inputs)  # 直接传递 inputs, 不需要 reshape
        outputs = outputs[:, -1, 0]
        predictions.extend(outputs.cpu().numpy())
        targets.extend(labels.cpu().numpy())

# 计算MSE
mse = mean_squared_error(targets, predictions)
print(f'Mean Squared Error: {mse}')
r_squared = r2_score(targets, predictions)
print(f'R-squared: {r_squared}')