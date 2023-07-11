#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   Config.py
@Desc         :   Configer of 'main process'
'''

import pickle
import os

class Config:

    def __init__(self, config_name=None):

        self.config_name = config_name
        self.api_header = 'apicall' # 数据中 api 所在列的名称
        self.pid_header = 'pid' # 数据种 pid 所在列的名称
        self.ppid_header = 'ppid' # 数据种 ppid 所在列的名称
        self.status_header = 'status'
        self.output_path = './outputs'

        self.name2time = self.load('exp_name2time')
        self.years_noise_data = self.load('years_noise_data')

        self.init_config()
        self.update_configer_dir()
        self.update_random_walk()
        if(self.embedding_model_name == 'w2v'):
            self.update_word2vec()
        elif('doc2vec' in self.embedding_model_name):
            self.update_doc2vec()
        if(self.pid_flag == 'node2vec'): self.update_node2vec()

        self.api_count_min = 3 # API 序列中最小包含的 API 数目
        self.sequence_length_min = 10 # 生成序列最小长度限制，用于筛选无效数据
        self.update_name()

    def load(self, name):
        with open(f'./outputs/{name}.pkl', 'rb') as f:
            data = pickle.load(f)
        return data

    def update_name(self):
        paramsinfo = self.config_name.split('_')[-1]
        status_info = 'status' if 'status' in paramsinfo else 'None'
        count_info = 'count' if 'count' in paramsinfo else 'None'
        filter_info = f'{self.api_count_min}, {self.sequence_length_min}'
        
        init_data_name = f'-{self.data_name}[({self.train_start_year}, {self.train_end_year}), ({self.test_start_year}, {self.test_end_year})]-{self.pid_flag}[{filter_info}]-{status_info}'
        init_data_name_all = f'-{self.data_name}[(2000, 2020), (2021, 2021)]-{self.pid_flag}[{filter_info}]-{status_info}'
        random_walk_name = f'-RW[{self.rw_k, self.rw_step}]-{count_info}'
        if(self.pid_flag == 'node2vec'): random_walk_name = f'-RW[{self.r, self.l, self.p, self.q}]-{count_info}'
        tovec_name = f'-{self.embedding_model_name}[{self.epochs}X{self.vector_size}+{self.window}]'
        todata_name = f'-{self.model_type}-{self.data_type}'

        self.init_data_name = '0-INIT' + init_data_name_all + '.pkl'
        self.random_walk_name = '1-RW' + init_data_name_all + random_walk_name + '.pkl'
        self.tovec_name = '2-Vec' + init_data_name + random_walk_name + tovec_name + '.pkl'
        self.todata_name = '3-Data'  + init_data_name + random_walk_name + tovec_name + todata_name + '.pkl'

    def init_config(self):
        self.train_start_year, self.train_end_year, self.test_start_year, self.test_end_year, self.data_name, self.pid_flag, self.embedding_model_name, self.model_type, self.model_name, self.data_type, self.params = self.config_name.split('_')
        self.train_start_year = int(self.train_start_year)
        self.train_end_year = int(self.train_end_year)
        self.test_start_year = int(self.test_start_year)
        self.test_end_year = int(self.test_end_year)
        self.params = self.params.split('+')
        if(self.model_name == 'biknn'): self.embedding_model_name = 'bi' + self.embedding_model_name

    def update_configer_dir(self):
        self.data_dir = f'./{self.data_name}'
        self.data_output_dir = f'./outputs/data'
        self.model_output_dir = f'./outputs/model'
        self.logger_output_dir = f'./outputs/logger'
        self.corpusinfo_output_dir = f'./outputs/corpusinfo'
        self.demo_output_dir = f'./outputs/demo'
        self.wrong_sample_output_dir = f'./outputs/wrong_sample'

        if not os.path.exists('./outputs'):
            os.mkdir('./outputs')
        
        for dir_path in [self.data_output_dir, self.model_output_dir, self.logger_output_dir, 
        self.corpusinfo_output_dir, self.demo_output_dir, self.wrong_sample_output_dir]:
            if not os.path.exists(dir_path): os.mkdir(dir_path)

    def update_random_walk(self):
        self.rw_k = 3
        self.rw_step = 100

    def update_doc2vec(self):
        self.vector_size = 64
        self.window = 5
        self.epochs = 10
        self.min_count = 1
        self.workers = 8

    def update_word2vec(self):
        self.vector_size = 64
        self.sg = 1 # skip-gram
        self.window = 5
        self.epochs = 10
        self.min_count = 1
        self.workers = 8
    
    def update_node2vec(self):
        self.p = 1
        self.q = 0.5
        self.r = 3
        self.l = 100

    def print_config(self):
        for item in self.__dict__.items():
            if(item[0] == 'name2time' or item[0] == 'years_noise_data'): continue
            print(item)