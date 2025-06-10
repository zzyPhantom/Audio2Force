from moviepy import VideoFileClip, AudioFileClip
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch, butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

import sounddevice as sd
import scipy.io.wavfile as wav

# print("Recording...")
# recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
# sd.wait()
# print("Recording complete.")

# wav.write("output.wav", sample_rate, recording)
# print("Saved as output.wav")

# ---------------------------------------
# 读刚录制的文件
fs, data = wav.read("audio_data.wav")
print(f"Sample rate: {fs}, Data shape: {data.shape}")
sample_rate = fs  # Hz
duration = data.shape[0] / fs  # seconds

# data如果是二维，转成一维
if len(data.shape) > 1:
    audio = data.mean(axis=1)
else:
    audio = data

# # Step 1: 加载视频并提取音频
# video_path = r"C:\Users\n11937386\OneDrive - Queensland University of Technology\program\FFT_test\TwoBatteries.mp4"  # 换成你的视频路径
# video = VideoFileClip(video_path)
# audio = video.audio
# sample_rate = audio.fps
# audio_array = audio.to_soundarray(fps=sample_rate)  # 转换为 numpy 数组
# audio = audio_array.mean(axis=1)  # 变成单通道

# 设计带通滤波器（例如 2kHz 到 5kHz）
lowcut = 100  # 高通截止
highcut = 10000  # 低通截止
low_limit = 0  # 显示设置
high_limit = 600  # 显示设置
segment_duration = 0.1  # 秒
start_time = 5  # 某一段时间的开始（秒）

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# # 某一段时间的PDS-S频谱图
# end_time = start_time + segment_duration
# start_sample = int(start_time * sample_rate)
# end_sample = int(end_time * sample_rate)
# audio_segment = audio[start_sample:end_sample]

# # 快速傅里叶变换
# N = len(audio_segment)              # 样本点数
# T = 1.0 / sample_rate               # 采样周期
# yf = fft(audio_segment)             # FFT
# xf = fftfreq(N, T)[:N//2]     # 频率轴

# psd = (1.0 / (sample_rate * N)) * (np.abs(yf[:N // 2]) ** 2) * 2

# # 找所有峰值
# peaks, _ = find_peaks(psd)
# peak_values = psd[peaks]

# # 找出前三大峰值
# top3_idx = np.argsort(peak_values)[-3:][::-1]  # 从大到小
# top3_freqs = xf[peaks[top3_idx]]
# top3_psd = peak_values[top3_idx]

# # 画图
# plt.figure(figsize=(12, 6))
# plt.plot(xf, psd, label='PSD')
# plt.plot(top3_freqs, top3_psd, 'ro', label='Top 3 Peaks')  # 标出前三个峰
# for i in range(3):
#     plt.text(top3_freqs[i], top3_psd[i], f'{top3_freqs[i]:.1f} Hz', fontsize=10, ha='center')

# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD (v²/Hz)')
# plt.title('Power Spectral Density: {}s to {}s, with Top 3 Peaks'.format(start_time, end_time))
# plt.grid(True)
# plt.legend()
# plt.xlim(0, 3500)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


total_duration = duration  # video.duration
num_segments = int(total_duration / segment_duration)

# 储存结果
times = []
top_freqs = []

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

    # 限制最大频率为 8.33Hz
    valid_idx = np.where(peak_freqs <= 600)[0]

    if len(valid_idx) == 0:
        continue  # 没有合适的峰值

    # 找最大峰值
    max_idx = valid_idx[np.argmax(peak_vals[valid_idx])]
    top_freq = peak_freqs[max_idx]

    # 记录时间和频率
    times.append(start_time + segment_duration / 2)
    top_freqs.append(top_freq)

# === Step 4: 绘图 ===
plt.figure(figsize=(12, 6))
plt.plot(times, top_freqs, 'o-', color='purple', label='Max Peak (<=500RPM)')

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Max Frequency Peak Over Time (≤ 500 RPM)')
plt.ylim(low_limit, high_limit)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()