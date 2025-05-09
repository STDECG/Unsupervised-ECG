import torch
import torch.nn as nn


# class DenseNet(nn.Module):
#     def __init__(self, growth_rate=8, input_channels=1, num_classes=2):
#         super(DenseNet, self).__init__()
#
#         self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=8, stride=1, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#
#         self.denseblock1 = self._make_dense_block(16, 6, growth_rate)
#         self.trans1 = self._make_transition_layer(16 + 6 * growth_rate, 48)
#
#         self.denseblock2 = self._make_dense_block(48, 4, growth_rate)
#         self.trans2 = self._make_transition_layer(48 + 4 * growth_rate, 80)
#
#         self.denseblock3 = self._make_dense_block(80, 6, growth_rate)
#
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(80 + 6 * growth_rate, num_classes)
#
#     def _make_dense_block(self, in_channels, num_layers, growth_rate):
#         layers = []
#         for i in range(num_layers):
#             layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
#         return nn.Sequential(*layers)
#
#     def _make_transition_layer(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.AvgPool1d(kernel_size=2, stride=2)
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.denseblock1(x)
#         x = self.trans1(x)
#
#         x = self.denseblock2(x)
#         x = self.trans2(x)
#
#         x = self.denseblock3(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#
# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, growth_rate):
#         super(DenseLayer, self).__init__()
#
#         self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(4 * growth_rate)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(growth_rate)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = torch.cat((x, out), 1)
#
#         return out


# class DenseNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(DenseNet, self).__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(2)
#         # 计算经过卷积和池化后的序列长度
#         new_length = 500
#         self.fc1 = nn.Linear(32 * new_length, 64)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#
#         return x


class DenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    input = torch.randn(size=(16, 1, 2000))
    model = DenseNet(num_classes=2)
    output = model(input)
    print(output.shape)
