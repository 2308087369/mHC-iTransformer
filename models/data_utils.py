import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end, 0:1] # Target is load_mw
        return seq_x, seq_y

def get_quarterly_data(file_path, seq_len=672, pred_len=672):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Feature columns
    cols = ['load_mw', 'temp_c', 'humidity_pct']
    data = df[cols].values
    
    quarters = []
    # Q1: Jan-Mar (Train), Apr (Test)
    # Q2: May-Jul (Train), Aug (Test)
    # Q3: Sep-Nov (Train), Dec (Test)
    
    month_splits = [
        ([1, 2, 3], [4]),   # Q1
        ([5, 6, 7], [8]),   # Q2
        ([9, 10, 11], [12]) # Q3
    ]
    
    for train_months, test_months in month_splits:
        train_mask = df.index.month.isin(train_months)
        test_mask = df.index.month.isin(test_months)
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        # Scale based on train data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        train_set = TimeSeriesDataset(train_scaled, seq_len, pred_len)
        test_set = TimeSeriesDataset(test_scaled, seq_len, pred_len)
        
        quarters.append({
            'train_loader': DataLoader(train_set, batch_size=32, shuffle=True),
            'test_loader': DataLoader(test_set, batch_size=1, shuffle=False),
            'scaler': scaler,
            'raw_test': test_data # For inverse scaling later if needed
        })
        
    return quarters
