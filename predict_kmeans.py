import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import densenet
from dataset import ECGDataset
from utils import set_seed


def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, label in tqdm(data_loader, desc="Extracting features"):
            x = model.features(data.to(device))
            x = model.avgpool(x)
            x = x.view(x.size(0), -1)

            features.append(x.cpu().numpy())
            labels.append(label.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def cluster_and_evaluate(features, true_labels, n_clusters=2):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)

    if np.mean(cluster_labels[:len(true_labels) // 2]) > np.mean(cluster_labels[len(true_labels) // 2:]):
        cluster_labels = 1 - cluster_labels

    acc = accuracy_score(true_labels, cluster_labels)
    f1 = f1_score(true_labels, cluster_labels)
    recall = recall_score(true_labels, cluster_labels)
    precision = precision_score(true_labels, cluster_labels)
    cm = confusion_matrix(true_labels, cluster_labels)

    print("\nClustering Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print("Confusion Matrix:")
    print(cm)

    visualize_features(features_scaled, true_labels, cluster_labels)

    return cluster_labels


def visualize_features(features, true_labels, cluster_labels):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=true_labels,
                    palette="viridis", alpha=0.7)
    plt.title("True Labels Visualization")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=cluster_labels,
                    palette="viridis", alpha=0.7)
    plt.title("Cluster Results Visualization")

    plt.tight_layout()
    plt.savefig("cluster_visualization.png")
    plt.show()


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = './data_labeled'
    test_dataset = ECGDataset(data_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = densenet.DenseNet(num_classes=2).to(device)
    model.load_state_dict(torch.load('./checkpoints/best-model-shdb.pt', map_location=device))

    features, true_labels = extract_features(model, test_loader, device)
    print(f"Extracted features shape: {features.shape}")

    cluster_labels = cluster_and_evaluate(features, true_labels, n_clusters=2)
    # np.save("extracted_features.npy", features)
    # np.save("cluster_results.npy", cluster_labels)
