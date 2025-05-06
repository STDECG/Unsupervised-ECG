import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import densenet
from dataset_shdb import ECGDataset
from utils import set_seed

import seaborn as sns


def evaluate(model, val_loader, device):
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            outputs = model(data.to(device))
            test_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            test_trues.extend(labels.detach().cpu().numpy())

    return test_preds, test_trues


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = './npy_files'
    test_dataset = ECGDataset(data_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = densenet.DenseNet(num_classes=2).to(device)
    model.load_state_dict(torch.load('./checkpoints/best-model-shdb.pt', map_location=device))

    test_preds, test_trues = evaluate(model, test_loader, device)
    test_acc = accuracy_score(test_trues, test_preds)
    test_f1 = f1_score(test_trues, test_preds)
    test_recall = recall_score(test_trues, test_preds)
    test_precision = precision_score(test_trues, test_preds)

    print(f'Acc: {test_acc}')
    print(f'F1: {test_f1}')
    print(f'Recall: {test_recall}')
    print(f'Precision: {test_precision}')

    # 计算混淆矩阵
    cm = confusion_matrix(test_trues, test_preds)

    # 定义TP、TN、FP、FN
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')

    ax.text(0.5, 0.5, f'{tn}\nTN', horizontalalignment='center', verticalalignment='center', color='black')
    ax.text(1.5, 0.5, f'{fp}\nFP', horizontalalignment='center', verticalalignment='center', color='black')
    ax.text(0.5, 1.5, f'{fn}\nFN', horizontalalignment='center', verticalalignment='center', color='black')
    ax.text(1.5, 1.5, f'{tp}\nTP', horizontalalignment='center', verticalalignment='center', color='black')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()