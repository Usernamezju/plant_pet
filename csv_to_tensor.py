# 文件名: csv_to_tensor.py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VoltageDataset(Dataset):
    def __init__(self, file_path, seq_len=100):
        # 1. 加载并标准化
        df = pd.read_csv(file_path)
        voltages = df.iloc[:, 2].values
        mean, std = np.mean(voltages), np.std(voltages)
        normalized = (voltages - mean) / (std + 1e-6)
        
        # 2. 切片
        num_samples = len(normalized) // seq_len
        self.data = torch.tensor(
            normalized[:num_samples * seq_len].reshape(num_samples, seq_len),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]