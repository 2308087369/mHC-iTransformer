import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import random

from models.models import LSTMModel, PatchTST, iTransformer
from models.mhc_itransformer import MHC_iTransformer
from models.time_filter import Model as TimeFilter
from models.DUET import DUETModel
from models.data_utils import data_provider

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

class Exp_Main:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'LSTM': LSTMModel,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'MHC_iTransformer': MHC_iTransformer,
            'TimeFilter': TimeFilter,
            'DUET': DUETModel,
        }
        
        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not supported. Choose from {list(model_dict.keys())}")
        
        print(f"Building model: {self.args.model}")
        
        # Initialize model based on type
        if self.args.model == 'LSTM':
            model = LSTMModel(
                input_dim=self.args.enc_in,
                hidden_dim=self.args.d_model,
                n_layers=self.args.e_layers,
                output_dim=self.args.c_out,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len
            )
        elif self.args.model == 'PatchTST':
            model = PatchTST(
                input_dim=self.args.enc_in,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                patch_len=self.args.patch_len,
                stride=self.args.stride,
                d_model=self.args.d_model,
                n_heads=self.args.n_heads,
                n_layers=self.args.e_layers,
                dropout=self.args.dropout
            )
        elif self.args.model == 'iTransformer':
            model = iTransformer(
                input_dim=self.args.enc_in,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                n_heads=self.args.n_heads,
                n_layers=self.args.e_layers,
                dropout=self.args.dropout
            )
        elif self.args.model == 'MHC_iTransformer':
            model = MHC_iTransformer(
                input_dim=self.args.enc_in,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                n_heads=self.args.n_heads,
                n_layers=self.args.e_layers,
                n_streams=self.args.n_streams,
                dropout=self.args.dropout
            )
        elif self.args.model == 'TimeFilter':
            # TimeFilter expects a config object
            model = TimeFilter(self.args)
        elif self.args.model == 'DUET':
            model = DUETModel(self.args)
        
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.model)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping_counter = 0
        best_valid_loss = float('inf')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Model forward
                if self.args.model in ['TimeFilter', 'DUET']:
                    # TimeFilter/DUET might return tuple (output, loss)
                    outputs, aux_loss = self.model(batch_x, None, is_training=True)
                else:
                    outputs = self.model(batch_x)
                    aux_loss = 0

                # Slice outputs to match prediction length
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = self.criterion(outputs, batch_y) + aux_loss
                train_loss.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            
            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.4f}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, self.criterion)
            test_loss = self.vali(test_loader, self.criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # Early stopping
            if vali_loss < best_valid_loss:
                best_valid_loss = vali_loss
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
                print("Validation loss decreased. Model saved.")
            else:
                early_stopping_counter += 1
                print(f"EarlyStopping counter: {early_stopping_counter} out of {self.args.patience}")
                if early_stopping_counter >= self.args.patience:
                    print("Early stopping.")
                    break

        # Load best model for testing
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['TimeFilter', 'DUET']:
                    outputs, _ = self.model(batch_x, None, is_training=False)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['TimeFilter', 'DUET']:
                    outputs, _ = self.model(batch_x, None, is_training=False)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        # Avoid division by zero
        range_y = np.max(trues) - np.min(trues)
        if range_y == 0:
            nrmse = 0
        else:
            nrmse = rmse / range_y

        print(f"Test Results - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, nRMSE: {nrmse:.4f}")
        
        # Save results
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        np.save(os.path.join(self.args.result_path, 'preds.npy'), preds)
        np.save(os.path.join(self.args.result_path, 'trues.npy'), trues)

        return mae, mse, rmse, nrmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # Basic Config
    parser.add_argument('--model', type=str, required=True, default='TimeFilter', help='model name')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    
    # Data Config
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/iTransformer_datasets/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--pca_dim', type=int, default=None, help='PCA dimension reduction')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting Task Config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # Model Config
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    # Specific Model Config
    parser.add_argument('--patch_len', type=int, default=16, help='patch length for PatchTST/TimeFilter')
    parser.add_argument('--stride', type=int, default=8, help='stride for PatchTST')
    parser.add_argument('--n_streams', type=int, default=4, help='number of streams for MHC')
    parser.add_argument('--alpha', type=float, default=0.1, help='TimeFilter alpha')
    parser.add_argument('--top_p', type=float, default=0.5, help='TimeFilter top_p')
    # DUET Config
    parser.add_argument('--noisy_gating', type=bool, default=True, help='DUET noisy gating')
    parser.add_argument('--num_experts', type=int, default=4, help='DUET num experts')
    parser.add_argument('--k', type=int, default=2, help='DUET k')
    parser.add_argument('--CI', type=int, default=1, help='DUET Channel Independence') 
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size for DUET encoder')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--result_path', type=str, default='results', help='path to save results')
    
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--pos', action='store_true', default=True, help='use positional embedding')

    # Optimization Config
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()
    
    # Small batch / debug mode adjustments if needed
    # (The user can just pass --batch_size 2 --train_epochs 1)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    set_seed(args.seed)
    
    exp = Exp_Main(args)
    
    print('>>>>>>> Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()
    
    print('>>>>>>> Start Testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.test()
