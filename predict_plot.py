import matplotlib.pyplot as plt
import torch

from densenet_ae_1d import Autoencoder
from utils import load_data


def plot_sing_lead(data, outputs):
    plt.figure(figsize=(12, 3), dpi=200)

    plt.plot(data, color='blue', label='Original')
    plt.plot(outputs, color='green', label='Generated')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = 'cpu' if torch.cuda.is_available() else 'cpu'

    test_file = './data_labeled/100.npy'

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('./checkpoints/best_model.pt', weights_only=True, map_location=device))

    data, label = load_data(test_file)

    data_expand = data.reshape(1, 1, -1)
    data_torch = torch.from_numpy(data_expand).float()

    model.eval()
    with torch.no_grad():
        outputs = model(data_torch)
        outputs = outputs.data.cpu().numpy()

    outputs = outputs.flatten()
    plot_sing_lead(data, outputs)
