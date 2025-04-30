import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(DenseLayer, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, 1, 0)
            self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(in_channels, 4 * growth_rate, 1, 1, 0)
            self.conv2 = nn.ConvTranspose2d(4 * growth_rate, growth_rate, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        bn1 = self.BN1(x)
        relu1 = self.relu1(bn1)
        conv1 = self.conv1(relu1)
        bn2 = self.BN2(conv1)
        relu2 = self.relu2(bn2)
        conv2 = self.conv2(relu2)
        return torch.cat([x, conv2], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(DenseBlock, self).__init__()
        self.DL1 = DenseLayer(in_channels + (growth_rate * 0), growth_rate, mode)
        self.DL2 = DenseLayer(in_channels + (growth_rate * 1), growth_rate, mode)
        self.DL3 = DenseLayer(in_channels + (growth_rate * 2), growth_rate, mode)

    def forward(self, x):
        DL1 = self.DL1(x)
        DL2 = self.DL2(DL1)
        DL3 = self.DL3(DL2)
        return DL3


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, c_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(TransitionBlock, self).__init__()
        out_channels = int(c_rate * in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        if mode == 'encode':
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            self.resize_layer = nn.AvgPool2d(2, 2)
        elif mode == 'decode':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0)
            self.resize_layer = nn.ConvTranspose2d(out_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        bn = self.BN(x)
        relu = self.relu(bn)
        conv = self.conv(relu)
        output = self.resize_layer(conv)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(3, 24, 3, 2, 1)
        self.BN1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU()
        self.db1 = DenseBlock(24, 8, 'encode')
        self.tb1 = TransitionBlock(48, 0.5, 'encode')
        self.db2 = DenseBlock(24, 8, 'encode')
        self.tb2 = TransitionBlock(48, 0.5, 'encode')
        self.db3 = DenseBlock(24, 8, 'encode')
        self.BN2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU()
        self.down_conv = nn.Conv2d(48, 24, 2, 2, 0)

    def forward(self, inputs):
        init_conv = self.init_conv(inputs)
        bn1 = self.BN1(init_conv)
        relu1 = self.relu1(bn1)
        db1 = self.db1(relu1)
        tb1 = self.tb1(db1)
        db2 = self.db2(tb1)
        tb2 = self.tb2(db2)
        db3 = self.db3(tb2)
        bn2 = self.BN2(db3)
        relu2 = self.relu2(bn2)
        down_conv = self.down_conv(relu2)
        return down_conv


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_conv = nn.ConvTranspose2d(24, 24, 2, 2, 0)
        self.db1 = DenseBlock(24, 8, 'decode')
        self.tb1 = TransitionBlock(48, 0.5, 'decode')
        self.db2 = DenseBlock(24, 8, 'decode')
        self.tb2 = TransitionBlock(48, 0.5, 'decode')
        self.db3 = DenseBlock(24, 8, 'decode')
        self.BN1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU()
        self.de_conv = nn.ConvTranspose2d(48, 24, 2, 2, 0)
        self.BN2 = nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU()
        self.out_conv = nn.ConvTranspose2d(24, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        up_conv = self.up_conv(inputs)
        db1 = self.db1(up_conv)
        tb1 = self.tb1(db1)
        db2 = self.db2(tb1)
        tb2 = self.tb2(db2)
        db3 = self.db3(tb2)
        bn1 = self.BN1(db3)
        relu1 = self.relu1(bn1)
        de_conv = self.de_conv(relu1)
        bn2 = self.BN2(de_conv)
        relu2 = self.relu2(bn2)
        out_conv = self.out_conv(relu2)
        output = self.tanh(out_conv)
        return output


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    ae = Autoencoder()
    inputs = torch.randn(size=(1, 3, 224, 224))
    print(ae(inputs).shape)  # torch.Size([1, 3, 224, 224])
    print(ae.encoder(inputs).shape)  # torch.Size([1, 24, 14, 14])
