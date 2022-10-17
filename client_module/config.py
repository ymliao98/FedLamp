from typing import List

class ClientAction:
    LOCAL_TRAINING = "local_training"

class ServerAction:
    LOCAL_TRAINING = "local_training"

class ClientConfig:
    def __init__(self,
                 idx: int,
                 master_ip: str,
                 master_port: int,
                 #action: str,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.master_ip = master_ip
        self.master_port = master_port
        #self.action = action
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.estimated_local_paras = dict()
        self.estimated_neighbor_paras = dict()
        self.bandwidth = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 6
        self.loss: float = 1
        self.running_time: int = 0
