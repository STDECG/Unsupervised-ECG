import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    def __init__(self, data_dir, train=True, test_size=0.2, random_state=42):
        super(ECGDataset, self).__init__()
        self.data_dir = data_dir
        full_data = np.load(os.path.join(self.data_dir, 'segments.npy'), allow_pickle=True)[()]
        full_label = np.load(os.path.join(self.data_dir, 'labels.npy'), allow_pickle=True)[()]
        train_data, test_data, train_label, test_label = train_test_split(
            full_data, full_label, test_size=test_size, random_state=random_state
        )
        if train:
            self.data = train_data
            self.label = train_label
        else:
            self.data = test_data
            self.label = test_label

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        data = data[np.newaxis, :]
        return torch.from_numpy(data).float(), torch.as_tensor(label).type(torch.LongTensor)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_path = './npy_files'

    train_set = ECGDataset(train_path)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    data, label = next(iter(train_loader))

    print(data.shape)
    print(label)
