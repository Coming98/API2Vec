#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   Config.py
@Desc         :   Configer of 'main process'
'''

import os
import utils

class Config:

    def __init__(self, config_name=None):

        self.api_header = 'apicall'
        self.pid_header = 'pid'
        self.ppid_header = 'ppid'
        self.status_header = 'status'

        self.api_count_min = 3
        self.sequence_length_min = 10

        self.output_path = './outputs'
        self.multi_classes = ['downloader', 'grayware', 'worm', 'backdoor']
        self.infos = utils.pickle_load('./outputs/infos.pkl')

        self.config_name = config_name

        self.init_config()
        self.update_configer_dir()
        self.update_random_walk()
        if(self.embedding_model_name == 'w2v'):
            self.update_word2vec()
        elif(self.embedding_model_name == 'doc2vec'):
            self.update_doc2vec()
        elif(self.embedding_model_name == 'bilstm'):
            self.update_bilstm()
        if(self.pid_flag == 'node2vec'): self.update_node2vec()

        self.update_name()

        if(self.task_name == 'attack'):
            self.update_attack()
        elif(self.task_name == 'target'):
            self.update_target()

    def update_name(self):
        paramsinfo = self.config_name.split('_')[-1]
        status_info = 'status' if 'status' in paramsinfo else 'None'
        count_info = 'count' if 'count' in paramsinfo else 'None'
        filter_info = f'{self.api_count_min}, {self.sequence_length_min}'
        
        init_data_name = f'-{self.data_name}-{self.pid_flag}[{filter_info}]-{status_info}'
        if(self.task_name == 'attack'):
            init_data_name = f'-attack-{self.data_name}-{self.pid_flag}[{filter_info}]-{status_info}'
        
        ramdom_walk_param_str = '-'.join(self.params[1:])
        random_walk_name = f'-{self.model_type}-RW[{self.rw_k, self.rw_step}]-{ramdom_walk_param_str}'
        if(self.pid_flag == 'node2vec'): random_walk_name = f'-{self.model_type}-RW[{self.r, self.l, self.p, self.q}]-{count_info}'
        tovec_name = f'-{self.embedding_model_name}[{self.epochs}X{self.vector_size}+{self.window}]'
        todata_name = f'-{self.model_type}-{self.data_type}'
        task_name = '-multi' if self.task_name == 'multi' else ''
        

        self.init_data_name = '0-INIT' + init_data_name + '.pkl'
        self.random_walk_name = '1-RW' + init_data_name + random_walk_name + '.pkl'
        self.tovec_name = '2-Vec' + init_data_name + random_walk_name + tovec_name + '.pkl'
        self.todata_name = '3-Data' + init_data_name + random_walk_name + tovec_name + todata_name + '.pkl'
        self.black_family_name = '4-Family' + init_data_name + random_walk_name + tovec_name + todata_name + '.pkl'

    def init_config(self):
        self.task_name, self.data_name, self.pid_flag, self.embedding_model_name, self.model_type, self.model_name, self.data_type, self.params = self.config_name.split('_')
        self.params = self.params.split('+')

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
        if(self.model_type == 'ml'):
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
        self.sg = 1 # CBOW
        self.window = 5
        self.epochs = 10
        self.min_count = 1
        self.workers = 8
    
    def update_bilstm(self):
        self.vector_size = 64
        self.sen_len = 1024
        self.epochs = 50
        self.batch_size = 64
        self.output_dim = 2
        self.rnn_size = 128
        self.rnn_layers = 3
        self.dropout = 0.5
        self.window = 5
        self.min_count = 1
        self.workers = 8

    def update_node2vec(self):
        self.p = 1
        self.q = 2
        self.r = 3
        self.l = 100


    def print_config(self):
        return
        for item in self.__dict__.items():
            if(item[0] in ['against_sample_names', 'attack_patterns', 'infos', 'ignored_names', 'test_names', 'type_names']): 
                print(f'({item[0]}, {len(item[1])})')
            else:
                print(item)

    def update_attack(self):
        self.against_sample_names = utils.pickle_load('./outputs/test_names.pkl')
        self.attack_patterns = utils.pickle_load('./Analysis/Against_Attack_Pattern/against_attack_pattern.pkl')
    
    def update_target(self):
        self.type_names = utils.pickle_load('./outputs/type_names.pkl')