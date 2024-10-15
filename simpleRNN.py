import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# 读取数据
data = pd.read_csv(r'D:\MoE\mixture-of-experts\data\tukeynew2_cleaned.csv')
# 假设数据中有一个 'date' 列作为时间戳
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

data = data[['IN_COD','IN_TP','IN_TN','IN_pH',
 		    'N-O-E_ORP','N-O-E_DO','N-O_NO',
 			'S-O-E_ORP','S-O-E_DO','S-O_NO','EF_COD']]
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

feature_names = ['IN_COD','IN_TP','IN_TN','IN_pH',
 		    'N-O-E_ORP','N-O-E_DO','N-O_NO',
 			'S-O-E_ORP','S-O-E_DO','S-O_NO','EF_COD']
reframed = series_to_supervised(data_scaled, 24, 1, feature_names=feature_names)
print(reframed.head())
# 在加载和预处理数据之后
values = reframed.values
n_total = values.shape[0]

# 定义测试集大小，例如最后10%的数据
# test_size = int(n_total * 0.1)
test_size = 1000
# 分割测试集
test = values[:test_size, :]
values = values[test_size:, :]  # 剩余的数据用于滚动交叉验证
n_total = values.shape[0]        # 更新总样本数

# 设置初始训练集大小和每次增加的大小
initial_train_size = int(n_total * 0.6)
expansion_size = int(n_total * 0.1)
validation_size = int(n_total * 0.1)

# 初始化索引
train_end = initial_train_size
validation_end = train_end + validation_size

iteration = 1  # 迭代计数器


# 定义特征数和历史步数
n_features = 11
n_obs = 24 * n_features

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleRNN(input_size=11, hidden_size=32, num_layers=2)
net.to(device)

# 滚动时间序列交叉验证
while validation_end <= n_total:
    print(f'\nIteration {iteration}: Training data from index 0 to {train_end}')
    print(f'Validation data from index {train_end} to {validation_end}')

    # 定义训练集和验证集
    train = values[:train_end, :]
    validation = values[train_end:validation_end, :]

    # 分割为输入和输出
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    validation_X, validation_y = validation[:, :n_obs], validation[:, -n_features]

    # 重塑为3D格式 [样本, 时间步, 特征]
    train_X = train_X.reshape((-1, 24, n_features))
    validation_X = validation_X.reshape((-1, 24, n_features))

    # 转换为PyTorch张量
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)
    validation_X_tensor = torch.tensor(validation_X, dtype=torch.float32)
    validation_y_tensor = torch.tensor(validation_y, dtype=torch.float32).view(-1, 1)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    validation_dataset = TensorDataset(validation_X_tensor, validation_y_tensor)

    train_loader = DataLoader(train_dataset, batch_size=144, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=144, shuffle=False)

    # 初始化模型（在循环内）
    net = SimpleRNN(input_size=11, hidden_size=32, num_layers=1)
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

    # 训练模型
    net.train()
    for epoch in range(5):  # 您可以根据需要调整epoch数量
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}')

        # 验证模型
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss / len(validation_loader)}')

    # 更新索引进行下一次迭代
    train_end += expansion_size
    validation_end = train_end + validation_size
    iteration += 1

print('Finished Training')

## 准备测试集数据
test_X, test_y = test[:, :n_obs], test[:, -n_features]
test_X = test_X.reshape((-1, 24, n_features))
test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32).view(-1, 1)

# 创建测试集数据加载器
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
test_loader = DataLoader(test_dataset, batch_size=144, shuffle=False)

net.eval()
predictions, targets = [], []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        predictions.extend(outputs.cpu().numpy().flatten())
        targets.extend(labels.numpy().flatten())

# 计算评估指标
mse = mean_squared_error(targets, predictions)
r_squared = r2_score(targets, predictions)
print(f'Train Mean Squared Error: {mse}')
print(f'Train R-squared: {r_squared}')
# 在测试集上评估模型
net.eval()
predictions, targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        predictions.extend(outputs.cpu().numpy().flatten())
        targets.extend(labels.numpy().flatten())

# 计算评估指标
mse = mean_squared_error(targets, predictions)
r_squared = r2_score(targets, predictions)
print(f'Test Mean Squared Error: {mse}')
print(f'Test R-squared: {r_squared}')

