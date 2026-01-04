import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from models.data_utils import get_quarterly_data
from models.models import PatchTST, iTransformer
from models.mhc_itransformer import MHC_iTransformer
import os

# Hyperparameters
SEQ_LEN = 672
PRED_LEN = 672
INPUT_DIM = 3
D_MODEL = 128
N_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    
    # Avoid division by zero
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    rmse = np.sqrt(np.mean((true - pred)**2))
    nrmse = rmse / (np.max(true) - np.min(true) + 1e-9)
    
    return mape, rmse, nrmse

def train_model(model, train_loader, model_name="Model"):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"\nStarting training for {model_name}...")
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {avg_loss:.6f}")
            
    return model

def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            # outputs: [B, pred_len, 1]
            predictions.append(outputs.cpu().numpy())
    return predictions[0].squeeze() # [pred_len]

def main():
    file_path = "datasets/aligned_15m_full.csv"
    quarters_data = get_quarterly_data(file_path, SEQ_LEN, PRED_LEN)
    
    # 对比第二季度 (Q2: 5-7月训练，8月测试)
    q_idx = 1
    q_data = quarters_data[q_idx]
    
    print(f"Experimental Target: Quarter {q_idx + 1}")
    train_loader = q_data['train_loader']
    test_loader = q_data['test_loader']
    scaler = q_data['scaler']
    
    # 获取 Ground Truth
    for batch_x, batch_y in test_loader:
        # batch_y: [B, pred_len, 1]
        true_y = batch_y.cpu().numpy()[0].squeeze() # [pred_len]
        break
    
    # 1. Baseline: PatchTST
    patch_tst = PatchTST(INPUT_DIM, SEQ_LEN, PRED_LEN, d_model=D_MODEL)
    train_model(patch_tst, train_loader, "PatchTST (Baseline)")
    patch_pred = predict(patch_tst, test_loader)
    
    # 2. Original iTransformer
    itrans_orig = iTransformer(INPUT_DIM, SEQ_LEN, PRED_LEN, d_model=D_MODEL)
    train_model(itrans_orig, train_loader, "iTransformer (Original)")
    itrans_orig_pred = predict(itrans_orig, test_loader)
    
    # 3. Ours: MHC-iTransformer
    mhc_itrans = MHC_iTransformer(INPUT_DIM, SEQ_LEN, PRED_LEN, d_model=D_MODEL)
    train_model(mhc_itrans, train_loader, "MHC-iTransformer (Ours)")
    mhc_itrans_pred = predict(mhc_itrans, test_loader)
    
    # 反归一化处理
    def inv_scale(pred, scaler):
        # pred is [pred_len]
        dummy = np.zeros((len(pred), 3))
        dummy[:, 0] = pred
        return scaler.inverse_transform(dummy)[:, 0]
    
    true_y_orig = inv_scale(true_y, scaler)
    patch_pred_orig = inv_scale(patch_pred, scaler)
    itrans_orig_pred_orig = inv_scale(itrans_orig_pred, scaler)
    mhc_itrans_pred_orig = inv_scale(mhc_itrans_pred, scaler)
    
    # 指标计算
    metrics_patch = calculate_metrics(true_y_orig, patch_pred_orig)
    metrics_itrans = calculate_metrics(true_y_orig, itrans_orig_pred_orig)
    metrics_mhc = calculate_metrics(true_y_orig, mhc_itrans_pred_orig)
    
    print("\n" + "="*75)
    print("EVALUATION RESULTS (First Test Week of Quarter 2)")
    print(f"{'Metric':<15} | {'PatchTST':<15} | {'iTransformer':<15} | {'MHC-iTransformer':<15}")
    print("-" * 75)
    print(f"{'MAPE (%)':<15} | {metrics_patch[0]:<15.4f} | {metrics_itrans[0]:<15.4f} | {metrics_mhc[0]:<15.4f}")
    print(f"{'RMSE':<15} | {metrics_patch[1]:<15.4f} | {metrics_itrans[1]:<15.4f} | {metrics_mhc[1]:<15.4f}")
    print(f"{'NRMSE':<15} | {metrics_patch[2]:<15.4f} | {metrics_itrans[2]:<15.4f} | {metrics_mhc[2]:<15.4f}")
    print("="*75)

    # 绘图
    plt.figure(figsize=(15, 8))
    plt.plot(true_y_orig, label='Ground Truth', color='black', linewidth=2, alpha=0.8)
    plt.plot(patch_pred_orig, label=f'PatchTST (MAPE: {metrics_patch[0]:.2f}%)', linestyle='--', alpha=0.6)
    plt.plot(itrans_orig_pred_orig, label=f'iTransformer (MAPE: {metrics_itrans[0]:.2f}%)', linestyle='-.', alpha=0.6)
    plt.plot(mhc_itrans_pred_orig, label=f'MHC-iTransformer (MAPE: {metrics_mhc[0]:.2f}%)', color='red', linewidth=2)
    
    plt.title(f"Quarter {q_idx + 1} Load Prediction Comparison")
    plt.xlabel("Time Steps (15 min intervals)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/q2_final_itransformer_comparison.png")
    print(f"\nFinal comparison plot saved to results/q2_final_itransformer_comparison.png")

if __name__ == "__main__":
    main()
