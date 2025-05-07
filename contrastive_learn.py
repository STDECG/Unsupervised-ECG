import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import scipy.io as sio

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)


# 读取MAT文件(支持v7.3格式)
def load_ecg_data(file_path, labeled_cnt=1000, af_cnt=500):
    """加载MAT格式的心电图数据，支持v7.3格式"""
    print(f"正在加载数据: {file_path}")

    try:
        # 尝试使用scipy.io.loadmat读取
        mat_data = sio.loadmat(file_path)
        # 假设数据在名为'traindata'的字段中 - 根据实际情况调整
        data_key = 'traindata'
        if data_key in mat_data:
            ecg_data = mat_data[data_key]
            print(f"已加载ECG数据，形状: {ecg_data.shape}")
        else:
            # 输出可用的字段名称
            print("MAT文件中的数据字段:")
            for key in mat_data.keys():
                if not key.startswith('__'):  # 跳过元数据
                    print(f"- {key}: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else '(未知)'}")
            print("未找到默认数据字段，将使用第一个非元数据字段")

            # 使用第一个非元数据字段
            for key in mat_data.keys():
                if not key.startswith('__'):
                    ecg_data = mat_data[key]
                    print(f"使用字段 '{key}'，形状: {ecg_data.shape}")
                    break
    except Exception as e:
        print(f"使用scipy.io.loadmat读取失败: {e}")
        print("尝试使用h5py读取...")

        try:
            # 尝试使用h5py读取MATLAB v7.3格式
            with h5py.File(file_path, 'r') as f:
                # 列出可用数据集
                print("MAT文件中的数据集:")
                for key in f.keys():
                    print(f"- {key}: {f[key].shape if hasattr(f[key], 'shape') else '(组)'}")

                # 使用第一个可用的数据集
                data_key = list(f.keys())[0]
                ecg_data = np.array(f[data_key])
                print(f"使用数据集 '{data_key}'，形状: {ecg_data.shape}")

                # 如果数据是存储为行优先，可能需要转置
                if ecg_data.shape[0] < ecg_data.shape[1]:
                    ecg_data = ecg_data.T
                    print(f"转置后形状: {ecg_data.shape}")
        except Exception as e2:
            print(f"使用h5py也失败: {e2}")
            return None, None, None

    # 处理数据类型
    ecg_data = ecg_data.astype(np.float32)

    # 确保维度正确（如果数据是多维的，尝试将其展平为二维）
    if len(ecg_data.shape) > 2:
        print(f"数据维度过高: {ecg_data.shape}，尝试展平")
        # 合并除了第一维外的所有维度
        new_shape = (ecg_data.shape[0], -1)
        ecg_data = ecg_data.reshape(new_shape)
        print(f"展平后形状: {ecg_data.shape}")

    # 划分数据
    total_samples = ecg_data.shape[0]
    print(f"总样本数: {total_samples}")

    # 确保有足够的样本
    if total_samples < labeled_cnt:
        print(f"警告：总样本数 {total_samples} 小于请求的有标签样本数 {labeled_cnt}")
        labeled_cnt = int(total_samples * 0.5)  # 使用一半样本作为有标签数据
        af_cnt = int(labeled_cnt * 0.5)  # 一半有标签样本作为房颤
        print(f"调整为：有标签样本 {labeled_cnt}，房颤样本 {af_cnt}")

    labeled_data = ecg_data[:labeled_cnt]  # 前labeled_cnt条有标签
    unlabeled_data = ecg_data[labeled_cnt:]  # 后面的无标签

    # 创建标签
    labels = np.zeros(labeled_cnt)
    labels[:af_cnt] = 1  # 前af_cnt条是房颤，标为1

    return labeled_data, labels, unlabeled_data


# 心电图数据集
class ECGDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ecg = self.data[idx].astype(np.float32)

        if self.transform:
            return self.transform(ecg)
        return ecg


# 对比学习的数据增强
class ECGContrastiveTransform:
    def __init__(self):
        # 心电图特定的数据增强方法
        pass

    def __call__(self, ecg):
        # 对同一条心电图产生两个不同的增强视角
        view1 = self.transform(ecg)
        view2 = self.transform(ecg)
        return view1, view2

    def transform(self, ecg):
        """单个心电图增强方法"""
        # 确保数据形状是一维的
        original_shape = ecg.shape
        if len(original_shape) > 1:
            ecg = ecg.reshape(-1)

        signal_length = len(ecg)

        # 1. 添加高斯噪声
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            ecg = ecg + noise_level * np.random.randn(*ecg.shape)

        # 2. 随机时间偏移
        if np.random.random() < 0.5:
            # 在±5%范围内偏移
            max_shift = int(signal_length * 0.05)
            shift = np.random.randint(-max_shift, max_shift) if max_shift > 0 else 0
            if shift > 0:
                ecg = np.pad(ecg, (0, shift), mode='constant')[shift:]
            elif shift < 0:
                ecg = np.pad(ecg, (-shift, 0), mode='constant')[:signal_length]

        # 3. 随机振幅缩放
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            ecg = ecg * scale

        # 4. 时间拉伸/压缩（在一定范围内）
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            new_length = int(signal_length * factor)
            if new_length < signal_length:
                # 压缩，然后填充
                indices = np.linspace(0, signal_length - 1, new_length, dtype=int)
                ecg_temp = ecg[indices]
                ecg = np.pad(ecg_temp, (0, signal_length - new_length), mode='constant')
            else:
                # 拉伸，然后截断
                indices = np.linspace(0, signal_length - 1, new_length, dtype=int)
                ecg = ecg[indices[:signal_length]]

        # 确保长度一致
        if len(ecg) != signal_length:
            ecg = np.resize(ecg, signal_length)

        # 恢复原始形状
        if len(original_shape) > 1:
            ecg = ecg.reshape(original_shape)

        return ecg


# CNN编码器网络
class ECGEncoder(nn.Module):
    def __init__(self, input_size=4000, embedding_dim=128):
        super(ECGEncoder, self).__init__()

        # 确保输入尺寸是已知的
        self.input_size = input_size

        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第二个卷积块
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第三个卷积块
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 第四个卷积块
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 计算卷积后的特征维度
        self.feature_dim = self._get_conv_output_dim(input_size)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def _get_conv_output_dim(self, input_size):
        """计算卷积层输出维度"""
        x = torch.zeros(1, 1, input_size)
        x = self.conv_layers(x)
        return x.numel() // x.size(0)

    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            # 添加通道维度 [batch, signal_length] -> [batch, 1, signal_length]
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] > 1 and x.shape[2] == 1:
            # 如果输入形状为 [batch, signal_length, 1]
            x = x.permute(0, 2, 1)  # 转换为 [batch, 1, signal_length]

        # 卷积特征提取
        x = self.conv_layers(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc(x)
        return x


# 对比学习模型
class ECGContrastiveModel(nn.Module):
    def __init__(self, encoder, projection_dim=64):
        super(ECGContrastiveModel, self).__init__()
        self.encoder = encoder

        # 投影头：将编码器输出映射到对比学习空间
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)  # 归一化
        return z

    def get_representation(self, x):
        """获取编码器输出（不经过投影头）"""
        return self.encoder(x)


# 对比损失函数（InfoNCE/NT-Xent）
def contrastive_loss(z1, z2, temperature=0.1):
    batch_size = z1.shape[0]

    # 特征拼接 [2*batch_size, projection_dim]
    features = torch.cat([z1, z2], dim=0)

    # 计算相似度矩阵
    similarity = torch.matmul(features, features.T) / temperature

    # 创建标签：正样本对索引
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ]).to(z1.device)

    # 移除对角线上的自相似度
    mask = torch.eye(2 * batch_size).bool().to(z1.device)
    similarity = similarity[~mask].view(2 * batch_size, -1)

    # 计算对比损失
    positives = torch.cat([
        similarity[torch.arange(2 * batch_size), labels],
    ]).view(2 * batch_size, 1)

    # 负对数似然
    logits = torch.cat([positives, similarity], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    loss = F.cross_entropy(logits, labels)
    return loss


# 训练函数
def train_contrastive(model, data_loader, optimizer, device, epochs=50):
    model.train()
    losses = []

    # 创建保存检查点的目录
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0
        # 使用tqdm显示每个epoch的进度条
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100)

        for batch_idx, (x1, x2) in enumerate(pbar):
            # 检查输入形状
            if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
                x1, x2 = x1.to(device), x2.to(device)

                # 前向传播
                z1 = model(x1)
                z2 = model(x2)

                # 计算损失
                loss = contrastive_loss(z1, z2)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新进度条显示当前batch的损失
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

                epoch_loss += loss.item()
            else:
                print(f"警告：遇到无效的批次数据类型: {type(x1)}, {type(x2)}")

        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"保存检查点到：{checkpoint_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('contrastive_loss.png')
    print("训练损失曲线已保存到 'contrastive_loss.png'")

    return model


# 提取特征并评估
def evaluate(model, labeled_data, labels, device):
    model.eval()

    # 构建数据集和数据加载器
    dataset = ECGDataset(labeled_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    # 使用tqdm显示特征提取的进度
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取特征", ncols=100):
            batch = batch.to(device)
            # 获取编码器输出（不经过投影头）
            feature = model.get_representation(batch)
            features.append(feature.cpu().numpy())

    features = np.vstack(features)

    # 使用K-means聚类，带进度信息
    print("执行K-means聚类...")
    kmeans = KMeans(n_clusters=2, random_state=42, verbose=1)
    predicted_clusters = kmeans.fit_predict(features)

    # 可能需要翻转预测标签以匹配真实标签
    # 房颤对应标签1，正常对应标签0
    if np.mean(predicted_clusters[:len(labels) // 2]) < np.mean(predicted_clusters[len(labels) // 2:]):
        predicted_clusters = 1 - predicted_clusters

    # 计算指标
    accuracy = accuracy_score(labels, predicted_clusters)
    conf_matrix = confusion_matrix(labels, predicted_clusters)
    report = classification_report(labels, predicted_clusters, target_names=['正常', '房颤'])

    print(f"准确率: {accuracy:.4f}")
    print(f"混淆矩阵:\n{conf_matrix}")
    print(f"分类报告:\n{report}")

    # 特征降维（为了可视化）
    print("执行特征可视化...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # 可视化特征
    plt.figure(figsize=(10, 8))
    af_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    plt.scatter(features_2d[normal_indices, 0], features_2d[normal_indices, 1], c='green', label='正常', alpha=0.5)
    plt.scatter(features_2d[af_indices, 0], features_2d[af_indices, 1], c='red', label='房颤', alpha=0.5)
    plt.title('特征空间可视化 (PCA降维)')
    plt.legend()
    plt.savefig('feature_visualization.png')
    print("特征可视化已保存到 'feature_visualization.png'")

    # 保存预测结果
    np.save('predicted_clusters.npy', predicted_clusters)
    print("预测结果已保存到 'predicted_clusters.npy'")

    return accuracy, conf_matrix, report


# 主函数
def main():
    data_file = './CPSC2025/traindata.mat'
    batch_size = 10
    epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据中...")
    labeled_data, labels, unlabeled_data = load_ecg_data(data_file)

    if labeled_data is None:
        print("数据加载失败，退出程序")
        return

    print(f"有标签数据: {labeled_data.shape}, 无标签数据: {unlabeled_data.shape}")

    # 确认信号长度
    signal_length = labeled_data.shape[1] if len(labeled_data.shape) > 1 else labeled_data.shape[0]
    print(f"信号长度: {signal_length}")

    # 创建数据集和数据加载器
    print("准备数据加载器...")
    transform = ECGContrastiveTransform()
    ecg_dataset = ECGDataset(unlabeled_data, transform=transform)
    data_loader = DataLoader(
        ecg_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 创建模型
    print("初始化模型...")
    encoder = ECGEncoder(input_size=signal_length).to(device)
    model = ECGContrastiveModel(encoder).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 训练模型
    print("开始对比学习训练...")
    model = train_contrastive(model, data_loader, optimizer, device, epochs=epochs)

    # 保存模型
    print("保存模型中...")
    torch.save(model.state_dict(), 'ecg_contrastive_model.pth')
    print("模型已保存到 'ecg_contrastive_model.pth'")

    # 评估模型
    print("评估模型性能...")
    evaluate(model, labeled_data, labels, device)

    print("所有步骤完成!")


if __name__ == "__main__":
    main()
