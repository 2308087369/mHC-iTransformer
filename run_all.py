import subprocess
import time
import os

# Defined datasets to run
datasets = [
    # (data_name, root_path, data_path, features_count, is_custom, freq)
    ('ETTh2', './datasets/iTransformer_datasets/ETT-small/', 'ETTh2.csv', 7, False, 'h'),
    ('ETTm1', './datasets/iTransformer_datasets/ETT-small/', 'ETTm1.csv', 7, False, 't'),
    ('ETTm2', './datasets/iTransformer_datasets/ETT-small/', 'ETTm2.csv', 7, False, 't'),
    ('Weather', './datasets/iTransformer_datasets/weather/', 'weather.csv', 21, True, 't'),
    ('Traffic', './datasets/iTransformer_datasets/traffic/', 'traffic.csv', 862, True, 'h'),
    ('Electricity', './datasets/iTransformer_datasets/electricity/', 'electricity.csv', 321, True, 'h'),
    ('Exchange', './datasets/iTransformer_datasets/exchange_rate/', 'exchange_rate.csv', 8, True, 'd'),
]

models = ['TimeFilter', 'iTransformer', 'MHC_iTransformer', 'PatchTST', 'LSTM']

results = {}

def run_training(model, data_name, root, path, dim, is_custom, freq, batch_size):
    data_arg = 'custom' if is_custom else data_name
    cmd = [
        'python', 'main.py',
        '--model', model,
        '--data', data_arg,
        '--root_path', root,
        '--data_path', path,
        '--batch_size', str(batch_size),
        '--train_epochs', '10',
        '--d_model', '128',
        '--e_layers', '2',
        '--learning_rate', '0.001',
        '--patch_len', '16',
        '--stride', '8',
        '--enc_in', str(dim),
        '--dec_in', str(dim),
        '--c_out', str(dim),
        '--freq', freq
    ]
    
    start_time = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if proc.returncode != 0:
            if "CUDA out of memory" in proc.stderr:
                return "OOM", proc.stderr
            return "Error", proc.stderr
        
        # Parse output
        mae = "N/A"
        mse = "N/A"
        output_lines = proc.stdout.split('\n')
        for line in output_lines:
            if "Test Results" in line:
                parts = line.split(',')
                mae = parts[0].split(':')[-1].strip()
                mse = parts[1].split(':')[-1].strip()
                return {'mae': mae, 'mse': mse, 'time': f"{end_time - start_time:.2f}s", 'batch_size': batch_size}, None
        
        return "No Output", proc.stdout
        
    except Exception as e:
        return "Exception", str(e)

print(f"{'Dataset':<15} | {'Model':<20} | {'MAE':<10} | {'MSE':<10} | {'Time':<10} | {'Batch':<5}")
print("-" * 80)

for data_name, root, path, dim, is_custom, freq in datasets:
    results[data_name] = {}
    for model in models:
        current_batch_size = 64
        while current_batch_size >= 1:
            print(f"Running {model} on {data_name} (BS={current_batch_size})...", end=' ', flush=True)
            res, error_log = run_training(model, data_name, root, path, dim, is_custom, freq, current_batch_size)
            
            if res == "OOM":
                print("OOM. Retrying with smaller batch size.")
                current_batch_size //= 2
                continue
            elif isinstance(res, dict):
                print(f"-> MAE: {res['mae']}, MSE: {res['mse']}")
                print(f"{data_name:<15} | {model:<20} | {res['mae']:<10} | {res['mse']:<10} | {res['time']:<10} | {res['batch_size']:<5}")
                results[data_name][model] = res
                break
            else:
                print(f"-> Failed: {res}")
                if res == "Error":
                     print(error_log)
                results[data_name][model] = res
                break

print("\n\nFinal Summary:")
print(f"{'Dataset':<15} | {'Model':<20} | {'MAE':<10} | {'MSE':<10} | {'Batch':<5}")
print("-" * 70)
for data_name, res in results.items():
    for model, metrics in res.items():
        if isinstance(metrics, dict):
            print(f"{data_name:<15} | {model:<20} | {metrics['mae']:<10} | {metrics['mse']:<10} | {metrics['batch_size']:<5}")
        else:
            print(f"{data_name:<15} | {model:<20} | {metrics:<10} | {metrics:<10} | {'-':<5}")
