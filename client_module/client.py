import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pulp import *
import random
from config import ClientConfig
from client_comm_utils import *
from training_utils import train, test
import utils
import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')

parser.add_argument('--dataset_type', type=str, default='MNIST')
parser.add_argument('--model_type', type=str, default='LR')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--local_iters', type=int, default=-1)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--adaptive', action="store_false", default=False)
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--algorithm', type=str, default='proposed')


args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx)) % 4 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

MODEL_SIZE = 14.0


def main():
    client_config = ClientConfig(
        idx=0,
        master_ip="",
        master_port=0
    )
    recorder = SummaryWriter("log_"+str(args.idx))
    # receive config
    master_socket = connect_get_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    #这里跟服务器通信然后获取配置文件，get_data_socket是堵塞的。
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)
    computation = client_config.custom["computation"]
    dynamics=client_config.custom["dynamics"]

    # init config
    args.local_ip = client_config.client_ip

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    # create model
    local_model = models.create_model_instance(args.dataset_type, args.model_type)
    #local_model.load_state_dict(client_config.para)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()


    print(args.batch_size, args.ratio)
    # create dataset
    print(len(client_config.custom["train_data_idxes"]))
    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=args.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    utils.count_dataset(train_loader)
    #test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    epoch_lr = args.lr
    local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
    old_para = copy.deepcopy(local_para)
    memory_para= torch.zeros_like(local_para)
    for epoch in range(1, 1+args.epoch):
        
        local_steps,compre_ratio=get_data_socket(master_socket)

        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        print("epoch-{} lr: {}, ratio: {} ".format(epoch, epoch_lr, args.ratio))
        print("local steps: ", local_steps)
        print("Compression Ratio: ", compre_ratio)

        #print("***")
        start_time = time.time()
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=args.model_type)
        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
        train_time = time.time() - start_time
        #train_time = train_time/local_steps
        train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        while train_time>10 or train_time<1:
             train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        train_time=train_time/10
        print("train time: ", train_time)

        #print(train_time/computation)
        test_loss, acc = test(local_model, test_loader, device, model_type=args.model_type)
        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
        
        print("send para")
        #compressed_paras = compress_model_top(local_para, compre_ratio)
        memory_para,compressed_paras = compress_model_top_with_memory(local_para,old_para,memory_para,compre_ratio)
        print("memory_para:",memory_para)
        start_time = time.time()
        send_data_socket(compressed_paras, master_socket)
        send_time=time.time()-start_time
        send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        while send_time>10 or send_time<1:
             send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        #compress_para=int(local_para.nelement()*compre_ratio)
        #send_time=send_time/compress_para
        print("send time: ",send_time)
        send_data_socket((train_time,send_time), master_socket)
        print("get begin")
        local_para = get_data_socket(master_socket)
        print("get")
        local_para.to(device)
        old_para=copy.deepcopy(local_para)
        #old_para.to(device)
        torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
        
    master_socket.shutdown(2)
    master_socket.close()

def compress_model_rand(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        select_n = int(send_para.nelement() * ratio)
        rd_seed = np.random.randint(0, np.iinfo(np.uint32).max)
        rng = np.random.RandomState(rd_seed)
        indices = rng.choice(send_para.nelement(), size=select_n, replace=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    return (select_para, select_n, rd_seed)

def compress_model_top(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)

    return (select_para, indices)

def compress_model_top_with_memory(local_para, old_para, memory_para,ratio):
    start_time = time.time()
    with torch.no_grad():
        print("local_para:",local_para)
        print("old_para:",old_para)
        local_para=local_para-old_para+memory_para
        print("local_para:",local_para)
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    restored_model = torch.zeros(send_para.nelement()).to(device)
    restored_model[indices] = select_para
    memory_para=local_para - restored_model
    model_size = select_para.nelement() * 4 / 1024 / 1024
    print("model_size:",model_size)
    print("memory_para:",memory_para)
    return (memory_para,(select_para, indices))

if __name__ == '__main__':
    main()
