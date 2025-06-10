from moviepy import VideoFileClip, AudioFileClip
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch, butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

import sounddevice as sd
import scipy.io.wavfile as wav

import pickle

# 参数
lowcut = 100  # 高通截止
highcut = 10000  # 低通截止
low_limit = 0  # 显示设置
high_limit = 600  # 显示设置
segment_duration = 0.1  # 秒
peak_freqs_high_limit = 600  # 峰值的最高Hz
peak_freqs_low_limit = 300  # 峰值的最低Hz
init_fequency = 550  # 默认频率

startTime = 12 # 开始时间
endTime = 110 # 结束时间
init_force = 1.4 # 初始力

# 读录制的文件
fs, data = wav.read("/home/zzy/Projects/Battery_sanding/data_test/audio_data.wav")
print(f"Sample rate: {fs}, Data shape: {data.shape}")
duration = data.shape[0] / fs  # seconds
sample_rate = fs  # Hz

# data如果是二维，转成一维
if len(data.shape) > 1:
    audio = data.mean(axis=1)
else:
    audio = data

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

num_segments = int(duration / segment_duration)

# 储存结果
times = []
top_freqs = []
top3_freqs = []

for i in range(num_segments):
    start_time = i * segment_duration
    end_time = start_time + segment_duration
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    if end_sample > len(audio):
        break

    audio_segment = audio[start_sample:end_sample]
    filtered_segment = bandpass_filter(audio_segment, lowcut, highcut, sample_rate)

    N = len(filtered_segment)
    T = 1.0 / sample_rate
    yf = fft(filtered_segment)
    xf = fftfreq(N, T)[:N // 2]
    psd = (1.0 / (sample_rate * N)) * (np.abs(yf[:N // 2]) ** 2) * 2

    # 找峰值
    peaks, _ = find_peaks(psd)
    peak_freqs = xf[peaks]
    peak_vals = psd[peaks]

    # 限制最大频率范围内的峰值索引
    valid_idx = np.where((peak_freqs >= peak_freqs_low_limit) & (peak_freqs <= peak_freqs_high_limit))[0]

    # 取有效峰值和频率
    valid_peak_vals = peak_vals[valid_idx]
    valid_peak_freqs = peak_freqs[valid_idx]

    if len(valid_peak_vals) == 0:
        top_freq = init_fequency  # 默认频率
        top3_freq = np.array([init_fequency, init_fequency, init_fequency])  # 统一填默认值
    else:
        max_idx_in_valid = np.argmax(valid_peak_vals)
        top_freq = valid_peak_freqs[max_idx_in_valid]

        # 找前三峰值（幅值最大的三个峰）
        if len(valid_peak_vals) >= 3:
            top3_idx_in_valid = np.argsort(valid_peak_vals)[-3:][::-1]  # 从大到小排序取前三
            top3_freq = valid_peak_freqs[top3_idx_in_valid]
        else:
            # 峰值不足3个时，补齐
            top3_freq = np.pad(valid_peak_freqs, (0, 3 - len(valid_peak_freqs)), mode='constant', constant_values=init_fequency)

    # 记录时间和频率
    times.append(start_time + segment_duration / 2)
    top_freqs.append(top_freq)
    top3_freqs.append(top3_freq)

times = np.array(times)
peak_vals_array = np.vstack(top3_freqs)
plot_peak_vals = np.array(top_freqs)  # 这里不需要vstack，直接转array

# 创建布尔掩码
mask = (times >= startTime) & (times <= endTime)

# 筛选出对应的输入特征
intime_peak_vals = peak_vals_array[mask]
plot_intime_peak_vals = plot_peak_vals[mask]

# （可选）也筛选对应的时间
intime_times = times[mask]

def load_data(filename):
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data

filename = '/home/zzy/Projects/Battery_sanding/data_test/force_pos_data.pkl'
data = load_data(filename)

# 提取前n秒的数据
times_pos_f = np.vstack([entry[0] for entry in data if entry[0] <= endTime and entry[0] >= startTime])
times_pos_f = times_pos_f.flatten()
force = np.vstack([entry[1] for entry in data if entry[0] <= endTime and entry[0] >= startTime])
pos = np.vstack([entry[5] for entry in data if startTime <= entry[0] <= endTime])
pos_z = np.array(pos[:,2]).reshape(-1,1)

# 初始化力数据
force = force - force[0] - init_force

aligned_forces = []
aligned_pos = []

for t in intime_times:
    idx = np.argmin(np.abs(times_pos_f - t))  # 找到与t最接近的力的时间点索引
    aligned_forces.append(force[idx])
    aligned_pos.append(pos_z[idx])

aligned_forces = np.array(aligned_forces)
aligned_pos = np.array(aligned_pos)


# 差分
delta_pos = np.diff(aligned_pos.reshape(-1))      # 长度 N-1
delta_time = np.diff(intime_times)    # 长度 N-1
delta_time[delta_time == 0] = 1e-6

# 速度
velocity = delta_pos / delta_time  # 长度 N-1
velocity = np.insert(velocity, 0, 0.0)  # 形状 (N,)
velocity = np.array(velocity.reshape(-1,1))

# 检查长度是否匹配
assert velocity.shape[0] == intime_peak_vals.shape[0], "样本数量不一致，无法拼接"

# 拼接成输入特征矩阵 (N, 4)
input_features = np.hstack([intime_peak_vals, velocity])
print(input_features.shape, aligned_forces.shape)

np.savez('/home/zzy/Projects/Battery_sanding/data_test/test_data.npz', x=input_features, y=aligned_forces)


# 创建图表和子图
fig, axes = plt.subplots(1, 3, figsize=(12, 10))
fig.suptitle("Time-Force and Time-Position Plots")

# 子图1: 时间-力
axes[0].plot(intime_times, aligned_forces, label="Force on Z-Axis", color="b")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Z-Axis Force (N)")
axes[0].set_title("Time vs Force")
axes[0].legend()
axes[0].grid(True)

# 子图2: 时间-末端位置误差
axes[1].plot(intime_times, aligned_pos, label="Pos_Z", color="g")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Z-Axis Position (m)")
axes[1].set_title("Time vs Z-Axis Position Error")
axes[1].grid(True)

axes[2].plot(intime_times, plot_intime_peak_vals, label="Frequency", color="r")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")
axes[2].set_title("Time vs Peak Frequency")
axes[2].grid(True)

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
