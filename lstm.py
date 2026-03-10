import torch
import torch.nn as nn

class DeepDiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepDiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        # 假设 x_seq 形状为 [seq_len, batch_size, input_size]
        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)  
        
        # 【填空 1】：初始化状态
        # 提示：LSTM 必须维护两个状态。它们的形状应该完全一致，且通常初始化为 0。
        h = torch.zeros(batch_size, self.hidden_size) 
        c = torch.zeros(batch_size, self.hidden_size) 
        
        outputs = []
        
        for t in range(seq_len):
            # 取出当前时刻的输入
            x_t = x_seq[t] 
            
            # 【填空 2】：执行状态更新
            # 提示：LSTMCell 的输入是一个元组，返回值也是一个元组。
            # 请写出如何利用上一步的 (h, c) 和当前输入 x_t 得到新的状态。
            h, c = self.lstm_cell(x_t, (h, c))
            
            # 【填空 3】：计算当前步的输出
            # 提示：决定当前时刻预测结果的是哪一个状态？请将其通过线性层。
            out = self.fc(h) 
            outputs.append(out)
            
        # 【填空 4】：整理输出张量
        # 提示：将列表中的 Tensor 沿着时间轴堆叠，使形状回到 [seq_len, batch_size, output_size]
        final_output = torch.stack(outputs, dim=0)
        
        return final_output, (h, c)