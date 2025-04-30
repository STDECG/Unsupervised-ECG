import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import load_data

if __name__ == '__main__':
    data_path = './data_labeled/'

    train_path = './data/train/'
    test_path = './data/test/'
    for i in [train_path, test_path]:
        if not os.path.exists(i):
            os.makedirs(i, exist_ok=True)

    npy_files = [os.path.join(data_path, npy_file) for npy_file in os.listdir(data_path)]

    labels = []
    for npy_file in tqdm(npy_files):
        _, label = load_data(npy_file)
        labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(npy_files, labels, test_size=0.2, stratify=labels,
                                                        random_state=42)

    for file in tqdm(X_train):
        shutil.copy(file, os.path.join(train_path, os.path.split(file)[-1]))

    for file in tqdm(X_test):
        shutil.copy(file, os.path.join(test_path, os.path.split(file)[-1]))

    print(f'Finishing copy {len(X_train)} files to {train_path}, {len(X_test)} files to {test_path}')

