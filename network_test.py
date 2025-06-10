import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

time_interval = 0.1       # 每个数据点间隔0.1秒 

class ForcePredictGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=2, output_size=1):
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

def predict(model, X_test, device):
    model.eval()  # 切换到评估模式
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()  # 预测结果

    return y_pred

def plot_results(y_true, y_pred):

    num_points = len(y_true)  # 预测/真实的点数

    times = np.arange(num_points) * time_interval

    plt.figure(figsize=(10,5))
    plt.plot(times, y_true, label='True Force')
    plt.plot(times, y_pred, label='Predicted Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.title('True vs Predicted Force over Time')
    plt.legend()
    plt.show()

def create_sequences(inputs, targets, seq_len=10, step=1):
    X, y = [], []
    for i in range(0, len(inputs) - seq_len + 1, step):
        X.append(inputs[i:i+seq_len])
        y.append(targets[i+seq_len-1])
    return np.array(X), np.array(y)

def main():
    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载字典
    checkpoint = torch.load('/home/zzy/Projects/Battery_sanding/data_using/force_predict_gru.pth', map_location=device)
    seq_len = checkpoint['seq_len']
    step = checkpoint['step']

    # 加载模型
    model = ForcePredictGRU(
        input_size=4,
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_size=1
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
    model.eval()

    # 创建测试序列
    data = np.load('/home/zzy/Projects/Battery_sanding/data_test/test_data.npz')
    freq_vel_np = data['x']  # 输入特征
    force_np = data['y']  # 对应力值
    X_seq, y_seq = create_sequences(freq_vel_np, force_np, seq_len, step)

    stats = np.load('/home/zzy/Projects/Battery_sanding/data_using/normalization_stats.npz')
    X_mean = stats['X_mean']
    X_std = stats['X_std']
    y_mean = stats['y_mean']
    y_std = stats['y_std']

    # 输入标准化
    X_seq_norm = (X_seq - X_mean) / X_std

    y_pred_norm = predict(model, X_seq_norm, device)

    # 输出反标准化
    y_pred = y_pred_norm * y_std + y_mean

    plot_results(y_seq.reshape(-1), y_pred)

if __name__ == '__main__':
    main()