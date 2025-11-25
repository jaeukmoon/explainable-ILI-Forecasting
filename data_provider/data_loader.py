import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe, _add_missing_numbers
import warnings
from pathlib import Path
import scaler
from summarize_module.summarizer import Summarizer

warnings.filterwarnings('ignore')




class Dataset_Custom(Dataset):
    def __init__(self,args, root_path,news_data_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.args = args
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.news_data_path = news_data_path
        self._add_missing_numbers = _add_missing_numbers
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.summarizer = Summarizer(summarize_llm = self.args.summarize_llm, args=self.args)
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.5)
        num_test = int(len(df_raw) * 0.4)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0 or 1:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            if self.features == 'SS':
                df_data = df_raw[['YEAR', 'WEEK', self.target]]
            else:
                df_data = df_raw


        # if self.features == 'M' :
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]

        elif self.set_type == 2:
            if self.features == 'SS' or 'MS':
                df_data = df_raw[['YEAR','WEEK', self.target]]
            else:
                df_data = df_raw


        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     scaler.scaler.fit(train_data.values)
        #     data = scaler.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values


        # 뉴스 데이터를 저장할 빈 리스트 생성
        news_content_summary = []

        # YEAR와 WEEK에 맞게 뉴스 데이터를 불러와서 추가
        for _, row in df_data.iterrows():
            year = row['YEAR']
            week = row['WEEK']

            if self.args.ex_data == 'news':
                summary_filename = f"{year}{str(week).zfill(2)}_summary.txt"
                summary_file_path = os.path.join('./dataset/US_news/influenza_data/', summary_filename)  # 뉴스 파일 경로로 변경 필요
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r', encoding='utf-8') as file:
                        content_summary = file.read()
                else:
                    filename = f"{year}{str(week).zfill(2)}.txt"  # 파일명 생성 (예: 202040.txt)
                    file_path = os.path.join('./dataset/US_news/influenza_data/', filename)  # 뉴스 파일 경로로 변경 필요
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    with open(summary_file_path, 'w', encoding='utf-8') as file:
                        content_summary = self.summarizer.get_summary(content)
                        file.write(content_summary)
                    print("Summary added for",filename)
            elif self.args.ex_data == 'report':
                summary_filename = f"cdc_{year}{str(week).zfill(2)}_summary_GPT4_new.txt"
                summary_file_path = os.path.join('./dataset/cdc_txt_data_total/', summary_filename)
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r+', encoding='utf-8') as file:
                        content_summary = file.read()
                        updated_summary = self._add_missing_numbers(content_summary,is_summary=True)
                        if updated_summary != content_summary:  # 변경이 있을 경우 파일 업데이트
                            file.seek(0)
                            file.write(updated_summary)
                            file.truncate()
                            print("Summary file updated:", summary_filename)
                if not os.path.exists(summary_file_path) or len(content_summary) <500:
                    filename = f"cdc_{year}{str(week).zfill(2)}.txt"  # 파일명 생성 (예: 202040.txt)
                    file_path = os.path.join('./dataset/cdc_txt_data_total/', filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    with open(summary_file_path, 'w', encoding='utf-8') as file:
                        content_summary = self.summarizer.get_summary(content, year, week)
                        updated_summary = self._add_missing_numbers(content_summary,is_summary=True)
                        file.write(updated_summary)
                    print("Summary added for",filename)
            # if content_summary and content_summary is not None and content_summary != "" and self.summarizer.is_informative(content_summary):
            news_content_summary.append(content_summary)  # 뉴스 내용을 리스트에 추가

        # 뉴스 내용을 데이터프레임에 추가
        df_data['News_Content'] = news_content_summary
        data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2, 2:-1]
        self.data_y = data[border1:border2, 2:-1]
        self.news = data[border1:border2, [-1]]
        self.year = data[border1:border2, [0]]
        self.week = data[border1:border2, [1]]
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        start_year_x = self.year[s_begin:s_begin+1]
        start_week_x = self.week[s_begin:s_begin+1]
        news = self.news[s_begin:s_end]


        return start_year_x, start_week_x, seq_x, seq_y, news, feat_id

    def __len__(self):
        if self.set_type == 2:
            if self.features == 'SS' or 'MS':
                length = (len(self.data_x) - self.seq_len - self.pred_len + 1)
            elif self.features == 'MM':
                length = (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        else:
            length = (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        return length # len(self.data_x): 676, 201, 297

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.5)
        num_test = int(len(df_raw) * 0.4)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in # len(self.data_x): 676, 201, 297

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



