import pandas as pd
import os

datasets = {
    'ETTh1': 'datasets/iTransformer_datasets/ETT-small/ETTh1.csv',
    'ETTh2': 'datasets/iTransformer_datasets/ETT-small/ETTh2.csv',
    'ETTm1': 'datasets/iTransformer_datasets/ETT-small/ETTm1.csv',
    'ETTm2': 'datasets/iTransformer_datasets/ETT-small/ETTm2.csv',
    'Weather': 'datasets/iTransformer_datasets/weather/weather.csv',
    'Traffic': 'datasets/iTransformer_datasets/traffic/traffic.csv',
    'Electricity': 'datasets/iTransformer_datasets/electricity/electricity.csv',
    'Exchange': 'datasets/iTransformer_datasets/exchange_rate/exchange_rate.csv'
}

for name, path in datasets.items():
    try:
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=1)
            # Assuming first column is date
            num_features = len(df.columns) - 1
            print(f"{name}: {num_features}")
        else:
            print(f"{name}: File not found at {path}")
    except Exception as e:
        print(f"{name}: Error {e}")
