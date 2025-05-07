import os

import mat73
import numpy as np
from tqdm import tqdm

from utils import resample


def label_convert(i):
    if 0 <= i < 500:
        return 0
    elif 500 <= i < 1000:
        return 1
    return -1


if __name__ == '__main__':
    train_data_path = './CPSC2025/traindata.mat'
    train_data = mat73.loadmat(train_data_path)['traindata']

    all_processed_data = []
    for ecg in train_data:
        resampled = resample(ecg, sample_rate=400, resample_rate=200)
        all_processed_data.append(resampled)

    all_processed_data = np.concatenate(all_processed_data)
    global_mean = np.mean(all_processed_data)
    global_std = np.std(all_processed_data) + 1e-8

    labeled_path = './data_labeled'
    unlabeled_path = './data_unlabeled'
    os.makedirs(labeled_path, exist_ok=True)
    os.makedirs(unlabeled_path, exist_ok=True)

    for i in tqdm(range(len(train_data))):
        ecg_data = train_data[i]
        ecg_data = resample(ecg_data, sample_rate=400, resample_rate=200)

        ecg_data = (ecg_data - global_mean) / global_std

        label = label_convert(i)
        data_dict = {'data': ecg_data, 'label': label}

        if i < 1000:
            np.save(os.path.join(labeled_path, str(i) + '.npy'), data_dict, allow_pickle=True)
        else:
            np.save(os.path.join(unlabeled_path, str(i - 1000) + '.npy'), data_dict, allow_pickle=True)

    np.save('global_stats.npy', {'mean': global_mean, 'std': global_std})
    print(f"全局归一化参数: mean={global_mean:.4f}, std={global_std:.4f}")
