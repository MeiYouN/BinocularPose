from torch import nn


class PoseCorrection(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512):
        super().__init__()

        # 关节级特征提取
        self.joint_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)

        # 时序融合
        self.temporal_fusion = nn.LSTM(hidden_dim, hidden_dim * 2, batch_first=True)

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """
        x: (batch, seq, joints, 3)
        """
        batch, seq, joints, _ = x.shape

        joint_features, a = self.joint_encoder(x.view(batch * seq * joints, 1, -1))
        joint_features = joint_features.view(batch, seq, joints, -1)

        spatial_feat, _ = self.spatial_attn(
            joint_features.mean(1),
            joint_features.mean(1),
            joint_features.mean(1)
        )

        temporal_out, _ = self.temporal_fusion(spatial_feat)

        delta = self.predictor(temporal_out)
        return delta
