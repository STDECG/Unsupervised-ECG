import os

import mat73
import numpy as np
from tqdm import tqdm

from utils import min_max_normalize


def label_convert(i):
    if 0 <= i < 500:
        return 0
    elif 500 <= i < 1000:
        return 1
    return -1


if __name__ == '__main__':
    train_data_path = './CPSC2025/traindata.mat'
    train_data = mat73.loadmat(train_data_path)['traindata']

    labeled_path = './data_labeled'
    unlabeled_path = './data_unlabeled'
    if not os.path.exists(labeled_path):
        os.makedirs(labeled_path, exist_ok=True)

    if not os.path.exists(unlabeled_path):
        os.makedirs(unlabeled_path, exist_ok=True)

    for i in tqdm(range(len(train_data))):
        ecg_data = train_data[i]
        ecg_data= min_max_normalize(ecg_data)
        label = label_convert(i)

        data_dict = {'data': ecg_data,
                     'label': label}

        if i < 1000:
            np.save(os.path.join(labeled_path, str(i) + '.npy'), data_dict, allow_pickle=True)
        else:
            np.save(os.path.join(unlabeled_path, str(i-1000) + '.npy'), data_dict, allow_pickle=True)
