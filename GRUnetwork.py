import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob

#训练参数
seq_len=50
step=1
epochs=20
batch_size=32
hidden_size=256
num_layers=2
learning_rate=0.00001
train_ratio=0.8  # 训练集比例

class ForcePredictGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1):
        super(ForcePredictGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)  # 输入到隐藏层权重
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)      # 隐藏到隐藏层权重
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)      # 偏置设为0

        # 初始化全连接层权重
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def create_sequences(inputs, targets, seq_len=10, step=1):
    X, y = [], []
    for i in range(0, len(inputs) - seq_len + 1, step):
        X.append(inputs[i:i+seq_len])
        y.append(targets[i+seq_len-1])
    return np.array(X), np.array(y)

def data_normalization(train_data, train_labels, test_data, test_labels):
    # 计算训练集的均值和标准差（按最后一维，即特征维度）
    mean = np.mean(train_data, axis=(0, 1), keepdims=True)  # shape: (1, 1, 4)
    std = np.std(train_data, axis=(0, 1), keepdims=True)    # shape: (1, 1, 4)
    std[std == 0] = 1e-8

    y_mean = np.mean(train_labels)
    y_std = np.std(train_labels)
    y_std = y_std if y_std != 0 else 1e-8

    # 应用到训练集和测试集
    train_data_norm = (train_data - mean) / std
    test_data_norm = (test_data - mean) / std

    train_labels_norm = (train_labels - y_mean) / y_std
    test_labels_norm = (test_labels - y_mean) / y_std

    np.savez('/home/zzy/Projects/Battery_sanding/data_using/normalization_stats.npz',
             X_mean=mean, X_std=std, y_mean=y_mean, y_std=y_std)

    return train_data_norm, train_labels_norm, test_data_norm, test_labels_norm, y_mean, y_std

def process_individual_dataset(file_path, seq_len, step):
    data = np.load(file_path)
    X = data['x']
    y = data['y']

    X_seq, y_seq = create_sequences(X, y, seq_len, step)
    return X_seq, y_seq

def data_split(files, seq_len=50, step=1, train_ratio=0.8):
    # 加载并分割数据
    X_seq_all = []
    y_seq_all = []

    for file_path in files:
        X_seq, y_seq = process_individual_dataset(file_path, seq_len, step)
        X_seq_all.append(X_seq)
        y_seq_all.append(y_seq)

    # 拼接所有段落数据
    X_seq_total = np.concatenate(X_seq_all, axis=0)
    y_seq_total = np.concatenate(y_seq_all, axis=0)

    # 设置随机种子以打乱数据和标签
    np.random.seed(42)
    indices = np.arange(X_seq_total.shape[0])
    np.random.shuffle(indices)

    data = X_seq_total[indices]
    labels = y_seq_total[indices]

    # 分割训练集
    train_size = int(train_ratio * data.shape[0])

    train_data = data[:train_size]
    train_labels = labels[:train_size]

    test_data = data[train_size:]
    test_labels = labels[train_size:]

    return train_data, train_labels, test_data, test_labels

def train_model(X_seq_train, y_seq_train, X_seq_test, y_seq_test, y_mean, y_std, epochs=20, batch_size=32):
    
    print(f'训练序列形状: {X_seq_train.shape}, 标签形状: {y_seq_train.shape}')  

    # 转为 PyTorch tensor
    X_train_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_seq_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_seq_test, dtype=torch.float32)

    # 构造 dataset 和 dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = ForcePredictGRU()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # 评估在测试集上的表现
        model.eval()
        with torch.no_grad():
            test_output = model(X_test_tensor)
            test_loss = criterion(test_output, y_test_tensor).item()

            # 反标准化
            test_output_real = test_output.cpu().numpy() * y_std + y_mean
            y_test_real = y_test_tensor.cpu().numpy() * y_std + y_mean

            # 计算真实尺度的MSE
            mse_real = np.mean((test_output_real - y_test_real) ** 2)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {mse_real:.4f}")

    return model

def main():
    # 数据存储位置
    file_list = sorted(
        glob.glob("/home/zzy/Projects/Battery_sanding/data_using/train_data_*.npz"),
        key=lambda x: int(x.split("_")[-1].split(".")[0])  # 按数字顺序排序
    )

    train_data, train_labels, test_data, test_labels = data_split(file_list, seq_len=seq_len, step=step, train_ratio=train_ratio)
    X_train_norm, y_train_norm, X_test_norm, y_test_norm, y_mean, y_std = data_normalization(train_data, train_labels, test_data, test_labels)

    model = train_model(X_train_norm, y_train_norm, X_test_norm, y_test_norm, y_mean, y_std, epochs, batch_size)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'seq_len': seq_len,
        'step': step
    }, '/home/zzy/Projects/Battery_sanding/data_using/force_predict_gru.pth')

if __name__ == '__main__':
    main()

