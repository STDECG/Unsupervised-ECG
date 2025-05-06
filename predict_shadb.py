import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import densenet
from dataset_shdb import ECGDataset
from utils import set_seed


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
    print(f'Test Acc: {test_acc}')
