import torch
import torch.nn as nn

class PhysiologicalReLU(nn.Module):
    """
    生理启发式线性阈值编码器 (LTE)
    物理意义：模拟机械敏感离子通道在超阈值刺激下的线性响应
    """
    def __init__(self, num_features):
        super(PhysiologicalReLU, self).__init__()
        # 每一个类别（胁迫类型）都拥有独立的一组增益和阈值
        self.gain = nn.Parameter(torch.ones(num_features))
        self.threshold = nn.Parameter(torch.full((num_features,), -0.1))

    def forward(self, x):
        # 严谨的阈值开启逻辑：y = g * max(0, x - b)
        activated = torch.clamp(x - self.threshold, min=0)
        return self.gain * activated