import subprocess
import time
import os
import sys
import argparse
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Run depth comparison experiment')
parser.add_argument('--dataset', type=str, default='Electricity', help='Dataset to run: Electricity, ETTh2, etc. or ALL')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Dataset configuration
# (data_name, root_path, data_path, features_count, is_custom, freq, pca_dim)
datasets_config = {
    'Electricity': ('./datasets/iTransformer_datasets/electricity/', 'electricity.csv', 30, True, 'h', 30),
    'ETTh2': ('./datasets/iTransformer_datasets/ETT-small/', 'ETTh2.csv', 7, False, 'h', None),
    'Traffic': ('./datasets/iTransformer_datasets/traffic/', 'traffic.csv', 30, True, 'h', 30),
    'Weather': ('./datasets/iTransformer_datasets/weather/', 'weather.csv', 21, True, 't', None),
}

if args.dataset == 'ALL':
    target_datasets = ['Electricity', 'ETTh2', 'Traffic', 'Weather']
else:
    target_datasets = [args.dataset]

# Models to compare
models = ['iTransformer', 'MHC_iTransformer', 'AttnRes_iTransformer']
# Depths to test
depths = [2, 4, 6, 8]
# Optimizer (using Muon as it performed best in previous run)
optimizer = 'Muon'

results = {}

def run_training(model, optimizer, data_name, root, path, dim, is_custom, freq, e_layers, pca_dim=None):
    data_arg = 'custom' if is_custom else data_name
    result_path = f'./results/depth_exp/{data_name}/{model}_L{e_layers}'
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--data', data_arg,
        '--optimizer', optimizer,
        '--root_path', root,
        '--data_path', path,
        '--batch_size', '32',  # Smaller batch size for deeper models to avoid OOM
        '--train_epochs', '10',
        '--d_model', '128',
        '--e_layers', str(e_layers),
        '--learning_rate', '0.001',
        '--patch_len', '16',
        '--stride', '8',
        '--enc_in', str(dim),
        '--dec_in', str(dim),
        '--c_out', str(dim),
        '--freq', freq,
        '--result_path', result_path,
        '--use_gpu', 'True',
        '--gpu', '0'
    ]
    
    if pca_dim is not None:
        cmd.extend(['--pca_dim', str(pca_dim)])
    
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
        rmse = "N/A"
        nrmse = "N/A"
        
        output_lines = proc.stdout.split('\n')
        for line in output_lines:
            if "Test Results" in line:
                parts = line.split(',')
                mae = parts[0].split(':')[-1].strip()
                mse = parts[1].split(':')[-1].strip()
                rmse = parts[2].split(':')[-1].strip()
                nrmse = parts[3].split(':')[-1].strip()
                return {
                    'mae': mae, 'mse': mse, 'rmse': rmse, 'nrmse': nrmse, 
                    'time': f"{end_time - start_time:.2f}s",
                    'result_path': result_path
                }, None
        
        return "No Output", proc.stdout
        
    except Exception as e:
        return "Exception", str(e)

print(f"{'Dataset':<12} | {'Model':<20} | {'Layers':<6} | {'MAE':<10} | {'MSE':<10} | {'RMSE':<10} | {'Time':<10}")
print("-" * 100)

for data_name in target_datasets:
    if data_name not in datasets_config:
        print(f"Dataset {data_name} configuration not found. Skipping.")
        continue
        
    root, path, dim, is_custom, freq, pca_dim = datasets_config[data_name]
    
    for e_layers in depths:
        for model in models:
            print(f"Running {model} (L={e_layers}) on {data_name}...", end=' ', flush=True)
            res, error_log = run_training(model, optimizer, data_name, root, path, dim, is_custom, freq, e_layers, pca_dim)
            
            if isinstance(res, dict):
                print(f"Done. MSE: {res['mse']}")
                print(f"{data_name:<12} | {model:<20} | {e_layers:<6} | {res['mae']:<10} | {res['mse']:<10} | {res['rmse']:<10} | {res['time']:<10}")
            else:
                print(f"Failed: {res}")
                if res == "Error" or res == "Exception":
                     # Print first few lines of error log for debugging
                     print(f"Error details: {error_log[:500]}...")

print("\nExperiment Completed.")
