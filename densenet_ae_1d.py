import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, mode='encode'):
        super(DenseLayer, self).__init__()
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        if mode == 'encode':
            self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, 1, 1, 0)
            self.conv2 = nn.Conv1d(4 * growth_rate, growth_rate, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose1d(in_channels, 4 * growth_rate, 1, 1, 0)
            self.conv2 = nn.ConvTranspose1d(4 * growth_rate, growth_rate, 3, 1, 1)
        self.BN2 = nn.BatchNorm1d(4 * growth_rate)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.BN1(x))
        out = self.conv1(out)
        out = self.relu2(self.BN2(out))
        out = self.conv2(out)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, mode='encode'):
        super(DenseBlock, self).__init__()
        self.DL1 = DenseLayer(in_channels + growth_rate * 0, growth_rate, mode)
        self.DL2 = DenseLayer(in_channels + growth_rate * 1, growth_rate, mode)
        self.DL3 = DenseLayer(in_channels + growth_rate * 2, growth_rate, mode)

    def forward(self, x):
        x = self.DL1(x)
        x = self.DL2(x)
        x = self.DL3(x)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, c_rate, mode='encode'):
        super(TransitionBlock, self).__init__()
        out_channels = int(c_rate * in_channels)
        self.BN = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0)
        if mode == 'encode':
            self.resize_layer = nn.AvgPool1d(2, 2)
        elif mode == 'decode':
            self.resize_layer = nn.ConvTranspose1d(out_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        x = self.relu(self.BN(x))
        x = self.conv(x)
        return self.resize_layer(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv1d(1, 24, 3, 2, 1)
        self.BN1 = nn.BatchNorm1d(24)
        self.relu1 = nn.ReLU()
        self.db1 = DenseBlock(24, 8, 'encode')
        self.tb1 = TransitionBlock(48, 0.5, 'encode')
        self.db2 = DenseBlock(24, 8, 'encode')
        self.tb2 = TransitionBlock(48, 0.5, 'encode')
        self.db3 = DenseBlock(24, 8, 'encode')
        self.BN2 = nn.BatchNorm1d(48)
        self.relu2 = nn.ReLU()
        self.down_conv = nn.Conv1d(48, 24, 2, 2, 0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.relu1(self.BN1(x))
        x = self.db1(x)
        x = self.tb1(x)
        x = self.db2(x)
        x = self.tb2(x)
        x = self.db3(x)
        x = self.relu2(self.BN2(x))
        return self.down_conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_conv = nn.ConvTranspose1d(24, 24, 2, 2, 0)
        self.db1 = DenseBlock(24, 8, 'decode')
        self.tb1 = TransitionBlock(48, 0.5, 'decode')
        self.db2 = DenseBlock(24, 8, 'decode')
        self.tb2 = TransitionBlock(48, 0.5, 'decode')
        self.db3 = DenseBlock(24, 8, 'decode')
        self.BN1 = nn.BatchNorm1d(48)
        self.relu1 = nn.ReLU()
        self.de_conv = nn.ConvTranspose1d(48, 24, 2, 2, 0)
        self.BN2 = nn.BatchNorm1d(24)
        self.relu2 = nn.ReLU()
        self.out_conv = nn.ConvTranspose1d(24, 1, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.up_conv(x)
        x = self.db1(x)
        x = self.tb1(x)
        x = self.db2(x)
        x = self.tb2(x)
        x = self.db3(x)
        x = self.relu1(self.BN1(x))
        x = self.de_conv(x)
        x = self.relu2(self.BN2(x))
        return self.tanh(self.out_conv(x))


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


if __name__ == '__main__':
    ae = Autoencoder()
    inputs = torch.randn(1, 1, 4000)
    print(inputs.shape)
    encoded = ae.encoder(inputs)
    print(encoded.shape)
    output = ae(inputs)
    print(output.shape)
