import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.oscillations.osc_sim import generate_distributed_sources
from models.emg_noise.emg_sim import generate_emg_noise

def main():
    # 1. Загрузка прямой модели (forward model)
    fm_path = os.path.join(os.path.dirname(__file__), '..', '..', 'forward_model', 'forward_model.pkl')
    if not os.path.exists(fm_path):
        print(f"Error: Forward model not found at {fm_path}")
        print("Please run get_fm.py or create a dummy forward_model.pkl first.")
        return

    with open(fm_path, 'rb') as f:
        fm_data = pickle.load(f)

    leadfield = fm_data['leadfield']
    src_pos = fm_data['src_pos']

    # Параметры симуляции
    Fs = 250  # Частота дискретизации (Гц)
    Ts = 10   # Длительность симуляции (сек)
    N = int(Fs * Ts)

    Nsrc = 100       # Общее число мозговых источников
    Ndistr = 10      # Количество "целевых" источников мозга
    flanker = 1      # Фланкер для краевых эффектов (сек)

    N_emg = 20       # Количество источников миографического шума

    # 2. Генерация фонового мозга и целевых осцилляций
    print("Генерация мозговой активности...")
    X_s, X_bg, X_n, z, GA, S = generate_distributed_sources(
        G=leadfield, Nsrc=Nsrc, Ndistr=Ndistr, flanker=flanker, Ts=Ts, Fs=Fs
    )

    # 3. Генерация миографического шума (EMG)
    print("Генерация миографического шума...")
    X_emg, S_emg, GA_emg = generate_emg_noise(
        G=leadfield, src_pos=src_pos, N_emg=N_emg, N=N, Fs=Fs, superficial_ratio=0.8
    )

    # Масштабирование (настройка отношения сигнал/шум)
    # Здесь можно регулировать амплитуду EMG относительно мозга
    emg_multiplier = 2.0
    X_emg = X_emg * emg_multiplier

    white_noise_multiplier = 0.5
    X_n = X_n * white_noise_multiplier

    # 4. Смешивание сигналов
    # Чистый сигнал мозга = Целевые осцилляции + фоновая активность
    X_brain = X_s + X_bg

    # Зашумленный сигнал = Мозг + EMG + белый шум сенсоров
    X_mixed = X_brain + X_emg + X_n

    # 5. Визуализация результатов
    print("Построение визуализаций...")
    time = np.arange(N) / Fs

    # Выбираем один сенсор (например, первый) для визуализации
    sensor_idx = 0

    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # --- Временные ряды ---
    axs[0, 0].plot(time, X_brain[sensor_idx, :], color='blue')
    axs[0, 0].set_title("Чистый сигнал мозга (Sensor 0)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")

    axs[1, 0].plot(time, X_emg[sensor_idx, :], color='red')
    axs[1, 0].set_title("Миографический шум (Sensor 0)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Amplitude")

    axs[2, 0].plot(time, X_mixed[sensor_idx, :], color='purple')
    axs[2, 0].set_title("Смешанный зашумленный сигнал (Sensor 0)")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Amplitude")

    # --- Спектры мощности (PSD) ---
    f, Pxx_brain = welch(X_brain[sensor_idx, :], fs=Fs, nperseg=Fs*2)
    axs[0, 1].semilogy(f, Pxx_brain, color='blue')
    axs[0, 1].set_title("PSD чистого сигнала мозга")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Power")
    axs[0, 1].set_xlim(0, 100)

    f, Pxx_emg = welch(X_emg[sensor_idx, :], fs=Fs, nperseg=Fs*2)
    axs[1, 1].semilogy(f, Pxx_emg, color='red')
    axs[1, 1].set_title("PSD миографического шума")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Power")
    axs[1, 1].set_xlim(0, 100)

    f, Pxx_mixed = welch(X_mixed[sensor_idx, :], fs=Fs, nperseg=Fs*2)
    axs[2, 1].semilogy(f, Pxx_mixed, color='purple')
    axs[2, 1].set_title("PSD смешанного сигнала")
    axs[2, 1].set_xlabel("Frequency (Hz)")
    axs[2, 1].set_ylabel("Power")
    axs[2, 1].set_xlim(0, 100)

    plt.tight_layout()
    # Сохраняем график, чтобы не блокировать выполнение
    output_fig = os.path.join(os.path.dirname(__file__), 'simulation_results.png')
    plt.savefig(output_fig)
    print(f"Результаты сохранены в: {output_fig}")
    plt.close(fig)

if __name__ == "__main__":
    main()
