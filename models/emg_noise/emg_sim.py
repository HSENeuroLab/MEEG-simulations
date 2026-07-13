import numpy as np
from scipy.signal import lfilter

def generate_emg_noise(G, src_pos, N_emg, N, Fs, superficial_ratio=0.8):
    """
    Генерирует высокочастотный авторегрессионный (AR) EMG шум (20-100 Гц)
    и проецирует его на сенсоры. Источники для шума могут предпочтительно
    выбираться из "поверхностных" (краевых) диполей.

    Параметры:
    ----------
    G : numpy.ndarray
        Матрица свинцового поля (Leadfield). Размер (Nsens, 3 * Nsites).
    src_pos : numpy.ndarray
        Координаты источников. Размер (Nsites, 3).
    N_emg : int
        Количество источников миографического шума.
    N : int
        Количество временных отсчетов.
    Fs : float
        Частота дискретизации.
    superficial_ratio : float
        Доля источников EMG, которые будут гарантированно посажены на
        наиболее "поверхностные" диполи (ближе к черепу/коже).

    Возвращает:
    -------
    X_emg : numpy.ndarray
        Сенсорные данные миографического шума (Nsens, N).
    S_emg : numpy.ndarray
        Сами сигналы EMG в источниках (N_emg, N).
    GA_emg : numpy.ndarray
        Проекция активных EMG источников (Nsens, N_emg).
    """
    Gx = G[:, 0::3]
    Gy = G[:, 1::3]
    Gz = G[:, 2::3]
    Nsens, Nsites = Gx.shape

    # 1. Выбор поверхностных диполей
    # Оцениваем "поверхностность" расстоянием от центра масс мозга
    center = np.mean(src_pos, axis=0)
    distances = np.linalg.norm(src_pos - center, axis=1)

    # Сортируем индексы по убыванию расстояния от центра (самые дальние = поверхностные)
    sorted_indices = np.argsort(distances)[::-1]

    N_superficial = int(N_emg * superficial_ratio)
    N_random = N_emg - N_superficial

    # Выбираем поверхностные источники из топ-20% самых удаленных
    top_20_percent = int(Nsites * 0.2)
    pool_superficial = sorted_indices[:top_20_percent]

    chosen_superficial = np.random.choice(pool_superficial, N_superficial, replace=False)

    # Остальные выбираем случайно из оставшихся
    pool_remaining = np.setdiff1d(np.arange(Nsites), chosen_superficial)
    chosen_random = np.random.choice(pool_remaining, N_random, replace=False)

    src_inds = np.concatenate([chosen_superficial, chosen_random])

    # 2. Проекция источников со случайным направлением диполя
    GA_emg = np.zeros((Nsens, N_emg))
    for i in range(N_emg):
        src_idx = src_inds[i]
        r = np.random.rand(3)
        r = r / np.linalg.norm(r)
        GA_emg[:, i] = Gx[:, src_idx]*r[0] + Gy[:, src_idx]*r[1] + Gz[:, src_idx]*r[2]

    # 3. Генерация высокочастотного AR-шума (миография)
    # Используем простую AR(1) модель, фильтруем белый шум.
    # Чтобы получить высокочастотный спектр, используем отрицательный коэффициент a1
    # Это создаст пик мощности на высоких частотах (ближе к Fs/2)

    S_emg = np.zeros((N_emg, N))
    for k in range(N_emg):
        # Генерируем белый шум
        w = np.random.randn(N)
        # AR(1) фильтр: y[n] = a1 * y[n-1] + w[n]
        # Для высоких частот берем отрицательный коэффициент, например -0.8
        a1 = -0.8 - 0.1 * np.random.rand() # случайный коэффициент от -0.8 до -0.9
        a = [1, -a1]
        b = [1]

        # Пропускаем через AR фильтр
        ar_signal = lfilter(b, a, w)

        # Приводим к единичной дисперсии
        ar_signal = ar_signal / np.std(ar_signal)
        S_emg[k, :] = ar_signal

    # 4. Проецируем на сенсоры
    X_emg = GA_emg @ S_emg

    return X_emg, S_emg, GA_emg
