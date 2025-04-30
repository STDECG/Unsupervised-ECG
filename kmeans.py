import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from densenet_ae_1d import Autoencoder


def extract_features(autoencoder, dataloader, device='cpu', pool_type='mean'):
    autoencoder.eval()
    autoencoder.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch
                all_labels.extend(labels.numpy())
            else:
                inputs = batch

            inputs = inputs.to(device)

            encoded = autoencoder.encoder(inputs)

            if pool_type == 'mean':
                features = encoded.mean(dim=2)
            elif pool_type == 'max':
                features = encoded.max(dim=2)[0]

            all_features.append(features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None

    return features, labels


if __name__ == '__main__':
    data = torch.randn(100, 1, 4000)
    dataloader = DataLoader(data, batch_size=16, shuffle=False)

    autoencoder = Autoencoder()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features, _ = extract_features(autoencoder, dataloader, device=device)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(scaled_features)

    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters)
    plt.title("Clustering Results")
    plt.show()
