import pandas as pd
import numpy as np
from click.core import batch
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch_geometric.nn import PositionalEncoding

from plot3d import vis_plot

# 标准化函数
def standardize(data, mean, std):
    return (data - mean) / np.where(std == 0, 1e-10, std)

import torch
from torch.utils.data import Dataset, DataLoader

class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.reshape(-1, seq_length, 17, 4)[:,:,:,:3]  # 每帧分为17个部位，每个部位4个特征
        self.y = y.reshape(-1, seq_length, 17, 4)[:,:,:,:3]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


import torch
import torch.nn as nn
from copy import deepcopy


class MultiPointFilter(nn.Module):
    def __init__(self,
                 num_points=17,
                 d_model=64,
                 nhead=4,
                 num_layers=3,
                 window_size=30):
        """
        多目标轨迹平滑滤波器
        :param num_points: 需要同时处理的目标数量 (默认17)
        :其他参数: 与单点滤波器保持一致配置
        """
        super().__init__()
        self.num_points = num_points

        # 为每个目标创建独立的滤波器实例
        self.filters = nn.ModuleList([
            deepcopy(
                SinglePointFilter(
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    window_size=window_size
                )
            ) for _ in range(num_points)
        ])

    def forward(self, x, reset_flags=None):
        """
        输入:
            x - 当前帧的多个目标坐标 (batch_size, seq, num_points, 3)
            reset_flags - 各目标的重置标志 [可选, (batch_size, num_points)]
        输出:
            平滑后的坐标 (batch_size, num_points, 3)
        """
        batch_size = x.shape[0]
        outputs = []

        # 并行处理每个目标
        for point_idx in range(self.num_points):
            # 提取当前目标数据 (batch_size, 3)
            point_data = x[:, :, point_idx, :]

            # 处理重置逻辑
            if reset_flags is not None:
                for b in range(batch_size):
                    if reset_flags[b, point_idx]:
                        self.filters[point_idx].reset(b)

            # 单目标滤波 (自动处理窗口缓存)
            smoothed = self.filters[point_idx](point_data)  # (batch_size, seq, 3)
            outputs.append(smoothed)

        # 重组为多目标格式
        return torch.stack(outputs, dim=1)  # (batch_size, num_points, 3)

    def reset(self, batch_indices=None):
        """重置指定batch的缓存"""
        for filter in self.filters:
            filter.reset(batch_indices)


class SinglePointFilter(nn.Module):
    """改进后的单目标滤波器（支持批量处理）"""

    def __init__(self, d_model=64, nhead=4, num_layers=3, window_size=30):
        super().__init__()
        self.window_size = window_size

        # 共享组件
        self.embedding = nn.Linear(3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, window_size)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )

        # 缓存系统 (batch_idx -> (data_cache, counter))
        self.cache = {}

    def _update_cache(self, x, batch_idx):
        """维护每个batch独立的滑动窗口缓存"""
        if batch_idx not in self.cache:
            self.cache[batch_idx] = {
                'data': x.unsqueeze(0),  # (1, 3)
                'counter': 1
            }
        else:
            cache = self.cache[batch_idx]
            new_data = torch.cat([cache['data'], x.unsqueeze(0)], dim=0)
            if cache['counter'] >= self.window_size:
                new_data = new_data[-self.window_size:]
                new_counter = self.window_size
            else:
                new_counter = cache['counter'] + 1
            self.cache[batch_idx] = {
                'data': new_data,
                'counter': new_counter
            }

    def forward(self, x):
        """
        输入:
            x - 当前帧的单个目标坐标 (batch_size,seq, 3)
        输出:
            平滑后的坐标 (batch_size, 3)
        """
        batch_size = x.shape[0]
        outputs = []

        for b in range(batch_size):
            cache_data = x

            # 特征嵌入
            embedded = self.embedding(cache_data)  # (seq_len, d_model)
            embedded = self.pos_encoder(embedded)

            # 因果Transformer处理
            features = self.transformer(
                embedded.unsqueeze(1),  # (seq_len, 1, d_model)
                mask=self._generate_causal_mask(embedded.size(0)).to(x.device),
                is_causal=True
            )  # (seq_len, 1, d_model)

            # 取最新帧输出
            output = self.output(features[-1, 0])  # (3,)
            outputs.append(output)

        return torch.stack(outputs)  # (batch_size, 3)

    def reset(self, batch_indices=None):
        """重置指定batch的缓存"""
        if batch_indices is None:
            self.cache.clear()
        else:
            for idx in batch_indices:
                if idx in self.cache:
                    del self.cache[idx]

    def _generate_causal_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def create_sequences(data, seq_length=5):
    """创建时间序列样本"""
    sequences = []
    for i in range(len(data)-seq_length-1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)


seq_length = 30  # 使用5帧时间窗口
batch_size = 4

def load_dataset():
    # 读取数据
    noise_data = pd.read_csv(
        'D:\desktop\MouseWithoutBorders\BinocularPose\BinocularPose\\filter\dataset\\hrnet\\acting1.csv')
    true_data = pd.read_csv(
        'D:\desktop\MouseWithoutBorders\BinocularPose\BinocularPose\\filter\dataset\\vitpose\\acting1.csv')

    # 转换为numpy数组并确保对齐
    datalen = min(len(true_data), len(noise_data))
    X = noise_data.values.astype(np.float32)[:datalen]
    y = true_data.values.astype(np.float32)[:datalen]
    # 构造时间序列样本 (T, 17, 4)

    X_seq = create_sequences(X, seq_length)  # (N, 5, 17*4)
    y_seq = create_sequences(y, seq_length)

    # 重新划分数据集
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2)


    # 创建数据加载器
    train_dataset = PoseDataset(X_train, y_train)
    val_dataset = PoseDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, )
    val_loader = DataLoader(val_dataset, batch_size)


    return train_loader, val_loader


def train(train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiPointFilter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(100):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            print(inputs.shape, targets.shape)
            inputs = inputs.view(-1, seq_length, 17, 3)
            targets = targets.view(-1, seq_length, 17, 3)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)
            loss = criterion(outputs, targets[:,-1].view(batch_size, 17, 3))
            loss.backward()
            train_loss += loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets[:,-1].view(batch_size,17,3)).item()

        print(
            f'Epoch {epoch + 1:03d} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f}')
        torch.save(model,'model.pt')


def denoise_frame(noise_frame, model, device,X_mean, X_std, y_mean, y_std):
    # 标准化输入
    noise_frame = standardize(noise_frame.reshape(1, -1), X_mean, X_std)
    # 转换为模型输入格式
    input_tensor = torch.FloatTensor(noise_frame.reshape(1, 17, 4)).to(device)
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
    # 反标准化
    output = output.cpu().numpy().flatten()
    output = output * y_std + y_mean
    return output.reshape(17, 4)

def sliding_inference(model, sequence, window=16):
    preds = []
    for i in range(len(sequence)-window-1):
        with torch.no_grad():
            output = model(sequence[i:i+window])
            preds.append(output.cpu().numpy())
    return np.concatenate(preds)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./model.pt').to(device)

    noise_data = pd.read_csv(
        'D:\desktop\MouseWithoutBorders\BinocularPose\BinocularPose\\filter\dataset\\trans\walking3.csv')
    X = noise_data.values.astype(np.float32)

    plot_pose = vis_plot()
    for i in range(len(X) - 16 - 1):
        with torch.no_grad():
            input = X[i:i + 16]
            input = input.reshape(-1 ,16, 17, 4)[:,:,:,:3]
            input = torch.FloatTensor(input).to(device)
            output = model(input)
            # print(output.shape)
            plot_pose.show(output.cpu().numpy()[0])



if __name__ == '__main__':
    # test()
    train_loader, val_loader = load_dataset()
    train(train_loader, val_loader)