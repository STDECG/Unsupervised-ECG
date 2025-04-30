import os
import random

import matplotlib.pyplot as plt
import neurokit2 as nk

from utils import load_data

if __name__ == '__main__':
    data_path = './data_labeled/'
    npy_files = os.listdir(data_path)

    random.shuffle(npy_files)

    for npy_file in npy_files:
        data, label = load_data(os.path.join(data_path, npy_file))
        ecg_data = nk.ecg_clean(data, sampling_rate=400)
        _, rpeaks = nk.ecg_peaks(ecg_data, sampling_rate=400)
        rpeaks = rpeaks['ECG_R_Peaks']

        plt.figure(figsize=(10, 5))

        plt.plot(ecg_data)
        plt.plot(rpeaks, ecg_data[rpeaks], 'ro')
        plt.title(f'Label: {label}')
        plt.show()
