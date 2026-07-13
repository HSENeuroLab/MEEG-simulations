% Скрипт полного пайплайна для генерации, смешивания и визуализации
% чистого ЭЭГ сигнала и миографического шума.

% Добавляем пути к моделям
addpath('../../models/oscillations');
addpath('../../models/emg_noise');

% 1. Загрузка прямой модели (forward model)
fm_path = '../../forward_model/forward_model.mat';
if ~isfile(fm_path)
    error('Forward model not found at %s. Please run get_fm.py or create dummy model.', fm_path);
end
load(fm_path, 'leadfield', 'src_pos');

% Параметры симуляции
Fs = 250;  % Частота дискретизации (Гц)
Ts = 10;   % Длительность симуляции (сек)
N = Fs * Ts;

Nsrc = 100;       % Общее число мозговых источников
Ndistr = 10;      % Количество "целевых" источников мозга
flanker = 1;      % Фланкер для краевых эффектов (сек)

N_emg = 20;       % Количество источников миографического шума

% 2. Генерация фонового мозга и целевых осцилляций
disp('Генерация мозговой активности...');
[X_s, X_bg, X_n, z, GA, S] = osc_sim(leadfield, Nsrc, Ndistr, flanker, Ts, Fs);

% 3. Генерация миографического шума (EMG)
disp('Генерация миографического шума...');
[X_emg, S_emg, GA_emg] = emg_sim(leadfield, src_pos, N_emg, N, Fs, 0.8);

% Масштабирование (настройка отношения сигнал/шум)
emg_multiplier = 2.0;
X_emg = X_emg * emg_multiplier;

white_noise_multiplier = 0.5;
X_n = X_n * white_noise_multiplier;

% 4. Смешивание сигналов
% Чистый сигнал мозга = Целевые осцилляции + фоновая активность
X_brain = X_s + X_bg;

% Зашумленный сигнал = Мозг + EMG + белый шум сенсоров
X_mixed = X_brain + X_emg + X_n;

% 5. Визуализация результатов
disp('Построение визуализаций...');
time = (0:N-1) / Fs;

% Выбираем один сенсор (например, первый) для визуализации
sensor_idx = 1;

figure('Position', [100, 100, 1200, 800]);

% --- Временные ряды ---
subplot(3, 2, 1);
plot(time, X_brain(sensor_idx, :), 'b');
title('Чистый сигнал мозга (Sensor 1)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(3, 2, 3);
plot(time, X_emg(sensor_idx, :), 'r');
title('Миографический шум (Sensor 1)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(3, 2, 5);
plot(time, X_mixed(sensor_idx, :), 'Color', [0.5, 0, 0.5]);
title('Смешанный зашумленный сигнал (Sensor 1)');
xlabel('Time (s)'); ylabel('Amplitude');

% --- Спектры мощности (PSD) ---
% Используем pwelch для оценки спектра
window = round(Fs * 2);

[Pxx_brain, f] = pwelch(X_brain(sensor_idx, :), window, [], [], Fs);
subplot(3, 2, 2);
semilogy(f, Pxx_brain, 'b');
title('PSD чистого сигнала мозга');
xlabel('Frequency (Hz)'); ylabel('Power');
xlim([0 100]);

[Pxx_emg, f] = pwelch(X_emg(sensor_idx, :), window, [], [], Fs);
subplot(3, 2, 4);
semilogy(f, Pxx_emg, 'r');
title('PSD миографического шума');
xlabel('Frequency (Hz)'); ylabel('Power');
xlim([0 100]);

[Pxx_mixed, f] = pwelch(X_mixed(sensor_idx, :), window, [], [], Fs);
subplot(3, 2, 6);
semilogy(f, Pxx_mixed, 'Color', [0.5, 0, 0.5]);
title('PSD смешанного сигнала');
xlabel('Frequency (Hz)'); ylabel('Power');
xlim([0 100]);

% Сохраняем график
saveas(gcf, 'simulation_results_matlab.png');
disp('Результаты сохранены в: simulation_results_matlab.png');
