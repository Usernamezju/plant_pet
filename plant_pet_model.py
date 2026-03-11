import torch
import torch.nn as nn
import torch.nn.functional as F
# 从外部文件导入自定义算子，实现解耦
from PhysioActivations import PhysiologicalReLU 

class MultiModalEncoder(nn.Module):
    def __init__(self, time_series_dim, impedance_dim, hidden_dim, output_dim):
        super(MultiModalEncoder, self).__init__()
        
        # 1. 时序信号分支
        self.cnn1d = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.gru = nn.GRU(16, hidden_dim, batch_first=True)
        
        # 2. 阻抗谱分支
        self.impedance_mlp = nn.Sequential(
            nn.Linear(impedance_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. 注意力融合模块
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        
        # 4. 决策输出层
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # --- 解耦嵌入：实例化自定义算子 ---
        self.physio_relu = PhysiologicalReLU(num_features=output_dim)

    def forward(self, x_time, x_impedance):
        # 时序处理
        x_t = F.relu(self.cnn1d(x_time.unsqueeze(1))).transpose(1, 2)
        _, h_n = self.gru(x_t)
        v_time = h_n.squeeze(0)

        # 阻抗处理
        v_impedance = self.impedance_mlp(x_impedance)

        # 特征融合
        v_combined = torch.cat([v_time, v_impedance], dim=1)
        alphas = self.attention_net(v_combined)
        v_fused = alphas[:, 0:1] * v_time + alphas[:, 1:2] * v_impedance

        # --- 快速响应通道输出 ---
        logits = self.classifier(v_fused)
        # 调用导入的解耦模块
        final_output = self.physio_relu(logits)
        
        return final_output, alphas

# 测试代码
if __name__ == "__main__":
    model = MultiModalEncoder(1, 50, 64, 3)
    # 模拟数据流
    out, weights = model(torch.randn(8, 100), torch.randn(8, 50))
    print("模型输出（已通过解耦的生理阈值激活）:\n", out.shape)