import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from moe import MoE

# 读取数据
data = pd.read_csv(r'D:\MoE\mixture-of-experts\data\tukeynew2_cleaned.csv')
# 假设数据中有一个 'date' 列作为时间戳
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

data = data[['IN_COD','IN_TP','IN_TN','IN_pH',
 		    'N-O-E_ORP','N-O-E_DO','N-O_NO',
 			'S-O-E_ORP','S-O-E_DO','S-O_NO','9#TMP']]
# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, feature_names=None):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        if feature_names:
            names += [('%s(t-%d)' % (feature_names[j], i)) for j in range(n_vars)]
        else:
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if feature_names:
            if i == 0:
                names += [('%s(t)' % (feature_names[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (feature_names[j], i)) for j in range(n_vars)]
        else:
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # 把它们放在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # 去掉含 NaN 的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg
feature_names = ['IN_COD','IN_TP','IN_TN','IN_pH',
 		    'N-O-E_ORP','N-O-E_DO','N-O_NO',
 			'S-O-E_ORP','S-O-E_DO','S-O_NO','7#TMP']
reframed = series_to_supervised(data_scaled, 24, 1, feature_names=feature_names)
print(reframed.head())

values = reframed.values
n_train_hours = 10225*0.8  # 60% 的数据用于训练
n_validation_hours = 10225 * 0.2  # 20% 的数据用于验证
n_test_hours = len(values) - n_train_hours - n_validation_hours  # 剩余20%为测试

# 分割数据
train = values[:8180, :]
validation = values[8180:10225, :]
test = values[10226:, :]

n_features = 11
n_obs = 24 * n_features

# 分割为输入和输出
train_X, y_train = train[:, :n_obs], train[:, -n_features]
validation_X, y_validation = validation[:, :n_obs], validation[:, -n_features]
test_X, y_test = test[:, :n_obs], test[:, -n_features]

# 重塑为3D 格式 [样本, 时间步, 特征]
X_train = train_X.reshape((train_X.shape[0], 24, n_features))
X_validation = validation_X.reshape((validation_X.shape[0], 24, n_features))
X_test = test_X.reshape((test_X.shape[0], 24, n_features))

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=144, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=144, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=144, shuffle=False)


# 简易 RNN 模型定义
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          nonlinearity='tanh')

        # 初始化 RNN 层权重
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'weight_hh' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)

        # 添加额外的全连接层
        self.fc1 = nn.Linear(hidden_size, 20)
        self.fc2 = nn.Linear(20, 1)

        # 正则化在优化器中处理

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # 只使用序列的最后一个输出
        x = torch.tanh(self.fc1(x))  # 第一个全连接层使用 tanh 激活函数
        x = self.fc2(x)  # 输出层没有激活函数
        return x


# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleRNN(input_size=11, hidden_size=32, num_layers=1)
net.to(device)

# 损失函数和优化器，加入 L2 正则化
criterion = nn.L1Loss()  # Keras 中使用的是 MAE，即 L1 损失
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)  # 添加 L2 正则化作为 weight_decay


# 训练模型
net.train()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}')

    # 验证模型
    val_loss = 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Epoch {epoch+1}, Val Loss: {val_loss / len(validation_loader)}')

print('Finished Training')


# 性能评估
net.eval()
predictions, targets = [], []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        predictions.extend(outputs.cpu().numpy().flatten())
        targets.extend(labels.cpu().numpy().flatten())

mse = mean_squared_error(targets, predictions)
print(f'Mean Squared Error: {mse}')
r_squared = r2_score(targets, predictions)
print(f'R-squared: {r_squared}')

predictions, targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        predictions.extend(outputs.cpu().numpy().flatten())
        targets.extend(labels.cpu().numpy().flatten())
mse = mean_squared_error(targets, predictions)
print(f'Test Mean Squared Error: {mse}')
r_squared = r2_score(targets, predictions)
print(f'Test R-squared: {r_squared}')
# results_df = pd.DataFrame({
#     'Predicted': predictions,
#     'Actual': targets
# })
# # 将结果保存到CSV文件
# results_df.to_csv(r'D:\MoE\mixture-of-experts\data\Predictions_and_Labels.csv', index=False)
#
# print("Results saved to CSV file.")
#


# import matplotlib.pyplot as plt
#
# # 选择一部分数据进行可视化，例如前200个数据点
# data_to_visualize = 200
#
# # 重新加载训练集和测试集的预测值和真实标签，以便进行可视化
# predictions_train, targets_train = [], []
# predictions_test, targets_test = [], []
# net.eval()
#
# with torch.no_grad():
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         outputs = net(inputs)
#         predictions_train.extend(outputs.cpu().numpy().flatten())
#         targets_train.extend(labels.cpu().numpy().flatten())
#
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device)
#         outputs = net(inputs)
#         predictions_test.extend(outputs.cpu().numpy().flatten())
#         targets_test.extend(labels.cpu().numpy().flatten())
#
# # 绘制训练集的预测与真实数据
# plt.figure(figsize=(14, 7))
# plt.subplot(1, 2, 1)
# plt.plot(targets_train[:data_to_visualize], label='Actual')
# plt.plot(predictions_train[:data_to_visualize], label='Predicted')
# plt.title('Train Data: Actual vs. Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Normalized Value')
# plt.legend()
#
# # 绘制测试集的预测与真实数据
# plt.subplot(1, 2, 2)
# plt.plot(targets_test[:data_to_visualize], label='Actual')
# plt.plot(predictions_test[:data_to_visualize], label='Predicted')
# plt.title('Test Data: Actual vs. Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Normalized Value')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
