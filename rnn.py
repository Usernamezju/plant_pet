import torch
import torch.nn as nn

# 基础数据准备
idx_to_char = ['h', 'e', 'l', 'o']
char_to_idx = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
x_data = [0, 1, 2, 2] # "hell"
y_data = [1, 2, 2, 3] # "ello"
one_hot_lookup = torch.eye(4)
x_one_hot = one_hot_lookup[x_data]
y_true = torch.LongTensor(y_data)

class TinyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TinyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x_seq, h_prev):
        out_logits = []
        for x_t in x_seq:
            # 注意：x_t 需要用 .unsqueeze(0) 增加 Batch 维度
            h_prev = self.rnn_cell(x_t.unsqueeze(0), h_prev)
            # 将隐状态转为输出
            output = self.fc(h_prev)
            out_logits.append(output)
            
        return torch.cat(out_logits, dim=0), h_prev

# 实例化模型
model = TinyRNN(input_size=4, hidden_size=8, output_size=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 模拟训练步骤
for epoch in range(1):
    h_state = torch.zeros(1, 8)
    outputs, _ = model(x_one_hot, h_state)
    loss = criterion(outputs, y_true)

    # 【填空 4】：标准三连动作
    # 提示：1. 梯度归零； 2. 反向传播计算梯度； 3. 更新参数
    optimizer.zero_grad() # 1. 
    loss.backward() # 2.
    optimizer.step() # 3.