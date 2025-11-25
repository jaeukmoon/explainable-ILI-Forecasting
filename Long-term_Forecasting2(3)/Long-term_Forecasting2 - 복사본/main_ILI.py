from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test, test_vis
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.Transformer import Transformer
from models.LSTM import LSTM

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, default='min_LSTM')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--multimodal', type=int, default=0)

parser.add_argument('--root_path', type=str, default='./')
parser.add_argument('--data_path', type=str, default='ILI_region_feature_2020-2024.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--target', type=str, default='5')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--pred_len', type=int, default=5)
parser.add_argument('--label_len', type=int, default=0)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--backbone', type=str, default='gpt2')
parser.add_argument('--textmodel', type=str, default='gpt2')
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862) #
parser.add_argument('--c_out', type=int, default=862) #
parser.add_argument('--patch_size', type=int, default=2)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='LSTM')
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)

parser.add_argument('--task_name', type=str, required=False, default='short_term_forecast',
                    help='task name, options: [long_term_forecast, short_term_forecast, imputation, anomaly_detection, classification]')
parser.add_argument('--factor', type=int, default=5, help='attn factor')
parser.add_argument('--activation', type=str, default='gelu', help='activation function to use, options: [gelu, relu]')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')

args = parser.parse_args()

def train():

    SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
    }

    rmses = []
    pccs = []

    rmses_temp = []
    pccs_temp = []

    results = []
    min_rmse = float('inf')  # 최소 MSE를 저장할 변수, 무한대로 초기화
    best_epoch_result = {'mse': None, 'mae': None, 'epoch': None, 'model_state': None}  # 최적의 결과를 저장할 딕셔너리


    for ii in range(args.itr):

        setting = '{}_multimodal{}_{}%_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, args.multimodal, args.percent,args.backbone, args.textmodel,args.seq_len, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers,
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'
        # 데이터 처리하는 부분
        train_data, train_loader = data_provider(args, 'train') # train_data.data_y.shape [161, 7] # len(train_loader): 14
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')  # len(test_loader): 74
        # test_data_vis, test_loader_vis = data_provider(args, 'test')/

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        # 모델 불러오는 부분
        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        elif args.model == 'Transformer':
            model = Transformer(args, device)
            model.to(device)
        elif args.model == 'LSTM':
            model = LSTM(args, device)
            model.to(device)
        else:
            model = GPT4TS(args, device)
        # mse, mae = test(model, test_data, test_loader, args, device, ii)

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        # 학습의 기준(loss)을 정의하는 부분
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        # 실제 학습 수행
        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device) # torch.Size([16, 104, 1])

                batch_y = batch_y.float().to(device)


                outputs = model(batch_x, ii)

                outputs = outputs[:, -1:, :]
                batch_y = batch_y[:, -1:, :].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            # vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)

            rmse_temp, pcc_temp  = test(model, test_data, test_loader, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            rmses_temp.append(rmse_temp)
            pccs_temp.append(pcc_temp)
            results.append({
                'iteration': ii,
                'epoch': epoch,
                'rmse': rmse_temp,
                'pcc': pcc_temp
            })
            if rmse_temp < min_rmse:
                min_rmse = rmse_temp
                best_epoch_result = {'rmse': rmse_temp, 'pcc': pcc_temp, 'epoch': epoch, 'model_state': model.state_dict()}


            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            # early_stopping(vali_loss, model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

        # best_model_path = path + '/' + 'checkpoint.pth'
        # 최고의 모델 상태로 모델을 로드
        # model.load_state_dict(torch.load(best_model_path)) # 원래 이거였음
        # model.load_state_dict(best_epoch_result['model_state'])
        print("------------------------------------")
        rmse, pcc = test(model, test_data, test_loader, args, device, ii)
        rmses.append(rmse)
        pccs.append(pcc)

    # DataFrame을 생성하고 CSV 파일로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv('./results/'+setting+'.csv', index=False)
    print("Results saved to 'results.csv'.")

    # 평균 성능, 최고 성능 출력
    rmses = np.array(rmses)
    pccs = np.array(pccs)
    print("rmse_mean = {:.4f}, rmse_std = {:.4f}".format(np.mean(rmses), np.std(rmses)))
    print("pcc_mean = {:.4f}, pcc_std = {:.4f}".format(np.mean(pccs), np.std(pccs)))
    print(f"Best performance in iteration {ii}: Epoch {best_epoch_result['epoch']} with RMSE = {best_epoch_result['rmse']:.4f} and PCC = {best_epoch_result['pcc']:.4f}")
    model.load_state_dict(best_epoch_result['model_state'])

    # 제일 좋은 성능 저장
    best_model_path = os.path.join(args.checkpoints, setting) + '/' + 'checkpoint.pth'
    torch.save(best_epoch_result['model_state'],best_model_path)

    # 테스트 데이터에 대한 예측, 시각화 수행
    #test_vis(model, test_data, test_loader, args, setting, device, ii)

if __name__=='__main__':
    train()