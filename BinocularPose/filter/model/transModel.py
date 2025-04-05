import os

import numpy as np
import torch
import torch.nn as nn
from numpy import mean
from torch.utils.data import DataLoader
from tqdm import tqdm

from Net import PoseCorrection
from dataset import PoseDataset


class RealTimeCorrector:
    def __init__(self, model_path, window_size=15):
        self.model = PoseCorrection().load_state_dict(torch.load(model_path))
        self.buffer = []
        self.window_size = window_size
        self.conf_thresh = 0.8  # 置信度阈值

    def process_frame(self, observed_pose, confidence):
        """
        输入当前帧观测值:
        observed_pose: (13,3) numpy数组
        confidence: (13,) numpy数组 0.0~1.0
        返回校正后的姿态
        """
        # 维护滑动窗口
        self.buffer.append({
            'obs': observed_pose,
            'conf': confidence
        })
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # 当缓冲区满时开始预测
        if len(self.buffer) == self.window_size:
            # 准备输入数据
            noisy_seq = np.array([f['obs'] for f in self.buffer])  # (seq, j, 3)
            conf_seq = np.array([f['conf'] for f in self.buffer])  # (seq, j)

            # 转换为张量
            inputs = torch.FloatTensor(noisy_seq).unsqueeze(0).cuda()  # (1, seq, j, 3)
            conf = torch.FloatTensor(conf_seq).unsqueeze(0).cuda()  # (1, seq, j)

            # 预测当前帧
            with torch.no_grad():
                pred = self.model(inputs, conf).cpu().numpy()[0]  # (j,3)

            # 融合结果
            corrected = observed_pose.copy()
            low_conf_mask = confidence < self.conf_thresh
            corrected[low_conf_mask] = pred[low_conf_mask]

            return corrected
        return observed_pose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    from plot3d_2 import vis_plot
    pose_show = vis_plot()
    # model = PoseCorrection().to(device)
    # model.load_state_dict(torch.load('./best_model.pth'))
    model = torch.load('./best_model.pth')
    model.to(device)
    model.eval()

    test_dir = "D:\Desktop\EveryThing\WorkProject\BinocularPose\BinocularPose\\filter\model\\testdata"
    dataset = PoseDataset(test_dir,10,False)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in train_loader:
        input = data['inputs'].cuda()
        target = data['targets'][0].cpu().numpy()

        output = model(input)
        # print(target)
        # print(output)
        output_c = output[0].cpu().detach().numpy()
        print(output_c)

        pose_show.show(output_c, target)
        # time.sleep(0.033)


# 使用示例
def main_trans():
    # 加载真实数据 (示例数据)
    data_dir = "D:\Desktop\EveryThing\WorkProject\Data\s1-videos\\run\\testdata"

    is_noise = False
    # 准备数据集
    if is_noise:
        if not os.path.exists(data_dir+'\\catch.pth'):
            dataset = PoseDataset(data_dir,10,is_noise)
            torch.save(dataset, data_dir+'\\catch.pth')
        else:
            dataset = torch.load(data_dir+'\\catch.pth')
    else:
        dataset = PoseDataset(data_dir, 10, is_noise)

    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化模型
    model = PoseCorrection().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    all_loss_list = []
    best_loss = float('inf')
    # 训练循环
    for epoch in range(100):
        epoch_loss_list = []
        for batch in tqdm(train_loader):
            inputs = batch['inputs'].cuda()  # (b, seq, j, 3)
            target = batch['targets'].cuda()  # (b, j, 3)


            # 预测当前帧
            pred = model(inputs)

            # 计算损失
            loss = criterion(pred, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_list.append(loss.item())
        all_loss_list.append(epoch_loss_list)
        all_loss = np.array(all_loss_list)
        np.savetxt(r'./loss.txt', all_loss, delimiter=',')

        epoch_loss = mean(epoch_loss_list)
        print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}')
        if  epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model, 'best_model.pth')
            print("Saved new best model!")



if __name__ == '__main__':

    test()
    # main_trans()

