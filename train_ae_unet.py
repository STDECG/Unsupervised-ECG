import os

import torch
from torch import nn
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import ECGDataset
from unet import UNet
from utils import set_seed, EarlyStopping


def train(model, criterion, train_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        data, _ = batch

        optimizer.zero_grad()
        preds = model(data.to(device))
        loss = criterion(preds, data.to(device) )
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    return train_loss


def evaluate(model, criterion, valid_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            data, _ = batch

            preds = model(data.to(device))
            loss = criterion(preds, data.to(device))
            val_loss += loss.item()
    val_loss /= len(valid_loader)
    return val_loss


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    check_points_path = './checkpoints/'
    if not os.path.exists(check_points_path):
        os.makedirs(check_points_path, exist_ok=True)

    train_path = './data_unlabeled/'
    train_dataset = ECGDataset(train_path)
    m = len(train_dataset)
    train_data, valid_data = random_split(train_dataset, [m - int(0.2 * m), int(0.2 * m)],
                                          generator=torch.Generator().manual_seed(42))

    batch_size = 1024

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)

    epochs = 100
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-06)
    criterion = nn.MSELoss().to(device)
    early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.0001,
                                   path=os.path.join(check_points_path,
                                                     f'best_model-unet.pt'))

    for epoch in range(epochs):
        train_loss = train(model, criterion, train_loader, optimizer, device)
        val_loss = evaluate(model, criterion, valid_loader, device)

        print(f'Epoch {epoch + 1} -- Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
