import os
from typing import List
import paramiko
from scp import SCPClient
from torch.utils.tensorboard import SummaryWriter
from communication_module.comm_utils import *


class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self,
                 config,
                 common_config,
                 user_name,
                 para_nums
                 ):
        #这个config就是后面的client_config
        self.config = config
        self.common_config = common_config
        self.user_name = user_name
        self.idx = config.idx
        self.worker_time =0.0
        self.socket = None
        self.train_info = None
        self.para_nums=para_nums
        self.__start_local_worker_process()


    def __start_local_worker_process(self):
        python_path = '/opt/anaconda3/envs/pytorch/bin/python'
        # python_path = '/data/yxu/software/Anaconda/envs/torch1.6/bin/python'
        os.system('cd ' + os.getcwd() + '/client_module' + ';nohup  ' + python_path + ' -u client.py --master_ip ' 
                     + self.config.master_ip + ' --master_port ' + str(self.config.master_port)  + ' --idx ' + str(self.idx) 
                     + ' --dataset_type ' + str(self.common_config.dataset_type) +  ' --model_type ' + str(self.common_config.model_type) 
                     + ' --epoch ' + str(self.common_config.epoch) + ' --batch_size ' + str(self.common_config.batch_size) 
                     + ' --ratio ' + str(self.common_config.ratio) + ' --lr ' + str(self.common_config.lr) + ' --decay_rate ' + str(self.common_config.decay_rate)
                     + ' --algorithm ' + self.common_config.algorithm + ' --step_size ' + str(self.common_config.step_size) 
                     + ' > client_' + str(self.idx) + '_log.txt 2>&1 &')

        print("start process at ", self.user_name, ": ", self.config.client_ip)

    def send_data(self, data):
        send_data_socket(data, self.socket)

    def send_init_config(self):
        self.socket = connect_send_socket(self.config.master_ip, self.config.master_port)
        send_data_socket(self.config, self.socket)

    def get_config(self):
        self.train_info=get_data_socket(self.socket)


class CommonConfig:
    def __init__(self):
        self.recoder: SummaryWriter = SummaryWriter()

        self.dataset_type = 'FashionMNIST'
        self.model_type = 'AlexNet'
        self.use_cuda = True
        self.training_mode = 'local'

        self.epoch_start = 0
        self.epoch = 200

        self.batch_size = 64
        self.test_batch_size = 64

        self.lr = 0.1
        self.decay_rate = 0.97
        self.step_size = 1.0
        self.ratio = 1.0
        self.algorithm = "proposed"

        self.master_listen_port_base = 57300
        self.p2p_listen_port_base = 50000

        self.worker_list: List[Worker] = list()
        #这里用来存worker的


class ClientConfig:
    def __init__(self,
                 idx: int,
                 client_ip: str,
                 master_ip: str,
                 master_port: int,
                 custom: dict = dict()
                 ):
        #custom 表示邻居
        self.idx = idx
        self.client_ip = client_ip
        self.master_ip = master_ip
        self.master_port = master_port
        self.neighbor_paras = None
        self.neighbor_indices = None
        self.train_time=0
        self.send_time=0
        self.local_steps=20
        self.compre_ratio=1
        self.average_weight=0
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.running_time: int = 0
        
