function [X_emg, S_emg, GA_emg] = emg_sim(G, src_pos, N_emg, N, Fs, superficial_ratio)
% Генерирует высокочастотный авторегрессионный (AR) EMG шум (20-100 Гц)
% и проецирует его на сенсоры. Источники для шума могут предпочтительно
% выбираться из "поверхностных" (краевых) диполей.
%
% Параметры:
% ----------
% G : Матрица свинцового поля (Leadfield). Размер (Nsens, 3 * Nsites).
% src_pos : Координаты источников. Размер (Nsites, 3).
% N_emg : Количество источников миографического шума.
% N : Количество временных отсчетов.
% Fs : Частота дискретизации.
% superficial_ratio : Доля источников EMG, которые будут гарантированно посажены на
%                     наиболее "поверхностные" диполи (ближе к черепу/коже).

    if nargin < 6
        superficial_ratio = 0.8;
    end

    Gx = G(:, 1:3:end);
    Gy = G(:, 2:3:end);
    Gz = G(:, 3:3:end);
    [Nsens, Nsites] = size(Gx);

    % 1. Выбор поверхностных диполей
    % Оцениваем "поверхностность" расстоянием от центра масс мозга
    center = mean(src_pos, 1);
    distances = sqrt(sum(bsxfun(@minus, src_pos, center).^2, 2));

    % Сортируем индексы по убыванию расстояния от центра (самые дальние = поверхностные)
    [~, sorted_indices] = sort(distances, 'descend');

    N_superficial = floor(N_emg * superficial_ratio);
    N_random = N_emg - N_superficial;

    % Выбираем поверхностные источники из топ-20% самых удаленных
    top_20_percent = floor(Nsites * 0.2);
    pool_superficial = sorted_indices(1:top_20_percent);

    % Случайный выбор поверхностных
    perm_sup = randperm(length(pool_superficial));
    chosen_superficial = pool_superficial(perm_sup(1:N_superficial));

    % Остальные выбираем случайно из оставшихся
    pool_remaining = setdiff(1:Nsites, chosen_superficial);
    perm_rem = randperm(length(pool_remaining));
    chosen_random = pool_remaining(perm_rem(1:N_random));

    src_inds = [chosen_superficial; chosen_random'];

    % 2. Проекция источников со случайным направлением диполя
    GA_emg = zeros(Nsens, N_emg);
    for i = 1:N_emg
        src_idx = src_inds(i);
        r = rand(3, 1);
        r = r / norm(r);
        GA_emg(:, i) = Gx(:, src_idx)*r(1) + Gy(:, src_idx)*r(2) + Gz(:, src_idx)*r(3);
    end

    % 3. Генерация высокочастотного AR-шума (миография)
    S_emg = zeros(N_emg, N);
    for k = 1:N_emg
        w = randn(1, N);
        % Для высоких частот берем отрицательный коэффициент, например от -0.8 до -0.9
        a1 = -0.8 - 0.1 * rand();
        a = [1, -a1];
        b = 1;

        ar_signal = filter(b, a, w);

        % Приводим к единичной дисперсии
        ar_signal = ar_signal / std(ar_signal);
        S_emg(k, :) = ar_signal;
    end

    % 4. Проецируем на сенсоры
    X_emg = GA_emg * S_emg;
end
