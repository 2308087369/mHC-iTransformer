import subprocess
import time
import os
import matplotlib.pyplot as plt
import numpy as np

import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Run all experiments')
parser.add_argument('--dataset', type=str, default='Electricity', help='Dataset to run: Electricity, ETTh2, etc. or ALL')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Defined datasets to run
all_datasets = [
    # (data_name, root_path, data_path, features_count, is_custom, freq, pca_dim)
    ('ETTh2', './datasets/iTransformer_datasets/ETT-small/', 'ETTh2.csv', 7, False, 'h', None),
    ('ETTm1', './datasets/iTransformer_datasets/ETT-small/', 'ETTm1.csv', 7, False, 't', None),
    ('ETTm2', './datasets/iTransformer_datasets/ETT-small/', 'ETTm2.csv', 7, False, 't', None),
    ('Weather', './datasets/iTransformer_datasets/weather/', 'weather.csv', 21, True, 't', None),
    ('Traffic', './datasets/iTransformer_datasets/traffic/', 'traffic.csv', 30, True, 'h', 30),
    ('Electricity', './datasets/iTransformer_datasets/electricity/', 'electricity.csv', 30, True, 'h', 30),
    ('Exchange', './datasets/iTransformer_datasets/exchange_rate/', 'exchange_rate.csv', 8, True, 'd', None),
]

if args.dataset == 'ALL':
    datasets = all_datasets
else:
    datasets = [d for d in all_datasets if d[0].lower() == args.dataset.lower()]
    if not datasets:
        print(f"Dataset {args.dataset} not found. Running Electricity by default.")
        datasets = [d for d in all_datasets if d[0] == 'Electricity']

models = ['TimeFilter', 'iTransformer', 'MHC_iTransformer', 'PatchTST', 'LSTM', 'DUET']
optimizers = ['Adam', 'AdamW', 'Muon']

results = {}

def run_training(model, optimizer, data_name, root, path, dim, is_custom, freq, batch_size, pca_dim=None):
    data_arg = 'custom' if is_custom else data_name
    result_path = f'./results/{data_name}/{model}_{optimizer}'
    cmd = [
        'python', 'main.py',
        '--model', model,
        '--data', data_arg,
        '--optimizer', optimizer,
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
        '--freq', freq,
        '--result_path', result_path
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
        train_time = "N/A"
        infer_time = "N/A"
        
        output_lines = proc.stdout.split('\n')
        for line in output_lines:
            if "Training Time:" in line:
                train_time = line.split(':')[-1].strip()
            if "Inference Time:" in line:
                infer_time = line.split(':')[-1].strip()
            if "Test Results" in line:
                parts = line.split(',')
                mae = parts[0].split(':')[-1].strip()
                mse = parts[1].split(':')[-1].strip()
                rmse = parts[2].split(':')[-1].strip()
                nrmse = parts[3].split(':')[-1].strip()
                return {
                    'mae': mae, 'mse': mse, 'rmse': rmse, 'nrmse': nrmse, 
                    'train_time': train_time, 'infer_time': infer_time,
                    'time': f"{end_time - start_time:.2f}s", 'batch_size': batch_size,
                    'result_path': result_path
                }, None
        
        return "No Output", proc.stdout
        
    except Exception as e:
        return "Exception", str(e)

def plot_predictions(data_name, model_names, results):
    """
    Plots prediction vs ground truth for one sample sequence from the test set for all models.
    """
    plt.figure(figsize=(15, 6))
    
    # Load ground truth from the first successful model (they should be the same)
    trues = None
    for model in model_names:
        if model in results[data_name] and isinstance(results[data_name][model], dict):
            path = results[data_name][model]['result_path']
            try:
                trues = np.load(os.path.join(path, 'trues.npy'))
                break
            except:
                continue
    
    if trues is None:
        return

    # Pick a sample index (e.g., 0) and a channel index (e.g., -1 for OT)
    sample_idx = 0
    channel_idx = -1 
    
    plt.plot(trues[sample_idx, :, channel_idx], label='GroundTruth', linewidth=2, color='black')

    for model in model_names:
        if model in results[data_name] and isinstance(results[data_name][model], dict):
            path = results[data_name][model]['result_path']
            try:
                preds = np.load(os.path.join(path, 'preds.npy'))
                plt.plot(preds[sample_idx, :, channel_idx], label=model)
            except:
                pass
                
    plt.title(f'Prediction Comparison on {data_name} (Sample {sample_idx}, Channel {channel_idx})')
    plt.legend()
    plt.grid(True)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/{data_name}_prediction.png')
    plt.close()

def plot_metrics_comparison(results, models):
    """
    Bar chart comparison of different metrics for different models on different datasets.
    """
    datasets = list(results.keys())
    metrics = ['mae', 'mse', 'rmse', 'nrmse']
    
    for metric in metrics:
        plt.figure(figsize=(15, 8))
        x = np.arange(len(datasets))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            vals = []
            for data in datasets:
                if model in results[data] and isinstance(results[data][model], dict):
                    try:
                        vals.append(float(results[data][model][metric]))
                    except:
                        vals.append(0)
                else:
                    vals.append(0)
            
            plt.bar(x + i * width, vals, width, label=model)
            
        plt.xlabel('Datasets')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(x + width * (len(models) - 1) / 2, datasets)
        plt.legend()
        plt.grid(True, axis='y')
        if not os.path.exists('figures'):
            os.makedirs('figures')
        plt.savefig(f'figures/comparison_{metric}.png')
        plt.close()

print(f"{'Dataset':<15} | {'Model':<20} | {'Opt':<8} | {'MAE':<10} | {'MSE':<10} | {'RMSE':<10} | {'nRMSE':<10} | {'TrainT':<8} | {'InferT':<8} | {'Time':<10} | {'Batch':<5}")
print("-" * 140)

for data_name, root, path, dim, is_custom, freq, pca_dim in datasets:
    results[data_name] = {}
    for model in models:
        for optimizer in optimizers:
            model_key = f"{model}_{optimizer}"
            current_batch_size = 64
            while current_batch_size >= 1:
                print(f"Running {model} with {optimizer} on {data_name} (BS={current_batch_size})...", end=' ', flush=True)
                res, error_log = run_training(model, optimizer, data_name, root, path, dim, is_custom, freq, current_batch_size, pca_dim)
                
                if res == "OOM":
                    print("OOM. Retrying with smaller batch size.")
                    current_batch_size //= 2
                    continue
                elif isinstance(res, dict):
                    print(f"-> MAE: {res['mae']}, MSE: {res['mse']}, RMSE: {res['rmse']}, nRMSE: {res['nrmse']}")
                    print(f"{data_name:<15} | {model:<20} | {optimizer:<8} | {res['mae']:<10} | {res['mse']:<10} | {res['rmse']:<10} | {res['nrmse']:<10} | {res['train_time']:<8} | {res['infer_time']:<8} | {res['time']:<10} | {res['batch_size']:<5}")
                    results[data_name][model_key] = res
                    break
                else:
                    print(f"-> Failed: {res}")
                    if res == "Error":
                         print(error_log)
                    results[data_name][model_key] = res
                    break
    
    # Plot predictions for this dataset
    # plot_predictions(data_name, models, results) # Disable for now or update to handle model_key

# Plot overall metrics comparison
# plot_metrics_comparison(results, models) # Disable for now or update to handle model_key

print("\n\nFinal Summary:")
print(f"{'Dataset':<15} | {'Model':<20} | {'Opt':<8} | {'MAE':<10} | {'MSE':<10} | {'RMSE':<10} | {'nRMSE':<10} | {'Batch':<5}")
print("-" * 100)
for data_name, res_dict in results.items():
    for key, res in res_dict.items():
        if isinstance(res, dict):
             # key is model_optimizer
             parts = key.rsplit('_', 1)
             if len(parts) == 2:
                 m, opt = parts
             else:
                 m, opt = key, 'Unknown'
                 
             print(f"{data_name:<15} | {m:<20} | {opt:<8} | {res['mae']:<10} | {res['mse']:<10} | {res['rmse']:<10} | {res['nrmse']:<10} | {res['batch_size']:<5}")
        else:
             parts = key.rsplit('_', 1)
             if len(parts) == 2:
                 m, opt = parts
             else:
                 m, opt = key, 'Unknown'
             print(f"{data_name:<15} | {m:<20} | {opt:<8} | {str(res):<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<5}")
