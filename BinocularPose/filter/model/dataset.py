import os

import numpy as np
import torch


class SyntheticNoiseGenerator:
    def __init__(self, clean_data):
        """
        clean_data: (num_frames, 13, 3) 真实数据
        """
        self.clean_data = clean_data
        self.noise_params = {
            'position_noise': 0.05,  # 5cm位置噪声
            'occlusion_prob': 0.1,  # 10%遮挡概率
            'outlier_scale': 0.3  # 30cm异常值幅度
        }

    def add_noise(self, frame_idx, joint_mask=None):
        """
        生成带噪声的观测数据
        joint_mask: 指定需要加噪声的关节索引
        """
        clean_frame = self.clean_data[frame_idx]
        noisy_frame = clean_frame.copy()

        # 随机选择要干扰的关节
        if joint_mask is None:
            joint_mask = np.random.rand(13) < self.noise_params['occlusion_prob']

        # 添加高斯噪声
        position_noise = np.random.normal(
            scale=self.noise_params['position_noise'],
            size=(np.sum(joint_mask), 3))
        noisy_frame[joint_mask] += position_noise

        # 添加异常值
        outlier_mask = np.random.rand(np.sum(joint_mask)) < 0.3
        noisy_frame[joint_mask][outlier_mask] += \
            np.random.uniform(-1, 1, size=(np.sum(outlier_mask), 3)) * self.noise_params['outlier_scale']

        # 生成置信度(噪声越大置信度越低)
        confidence = np.ones(13)
        confidence[joint_mask] = 0.5 - 0.4 * (np.abs(position_noise).mean(axis=1) / 0.05)
        confidence = np.clip(confidence, 0.1, 0.9)

        return noisy_frame, confidence, joint_mask


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, gt_dir, seq_length=15, is_noise=True):
        self.seq_length = seq_length
        self.samples = []
        self.is_noise = is_noise

        gt_files = sorted([f for f in os.listdir(gt_dir) if f.startswith('test')])

        for g_file in gt_files:
            gt_data = np.loadtxt(os.path.join(gt_dir, g_file), delimiter=',', skiprows=1)  # (frames, 39)

            # 转换为(frames, 13, 3)
            gt_poses = gt_data.reshape(-1, 13, 3)
            noise_gen = SyntheticNoiseGenerator(gt_poses)

            # 生成滑动窗口样本
            for idx in range(len(gt_poses) - self.seq_length):
                noisy_seq = []
                for i in range(idx, idx + self.seq_length):
                    if self.is_noise:
                        noisy_frame, conf, joint_mask = noise_gen.add_noise(i)
                    else:
                        noisy_frame = gt_poses[i]
                    noisy_seq.append(noisy_frame)
                inputs = np.array(noisy_seq)
                target = gt_poses[idx + self.seq_length - 1]

                self.samples.append((
                    inputs.astype(np.float32),
                    target.astype(np.float32)
                ))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, targets = self.samples[idx]

        return {
            'inputs': torch.FloatTensor(inputs),
            'targets': torch.FloatTensor(targets),
        }
