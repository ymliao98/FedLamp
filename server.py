import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.nn.functional as F
from communication_module.comm_utils import *
from training_module import datasets, models, utils
from training_utils import test, train

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
#parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--model_type', type=str, default='VGG')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--algorithm', type=str, default='proposed')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--action_num', type=int, default=9)
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
SERVER_IP = "127.0.0.1"

global_is_end=0
def main():

    # init config
    common_config = CommonConfig()
    common_config.master_listen_port_base += random.randint(0, 20) * 20
    common_config.model_type = args.model_type
    common_config.batch_size = args.batch_size
    common_config.ratio = args.ratio
    common_config.epoch = args.epoch
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.algorithm = args.algorithm
    common_config.step_size = args.step_size

    avg_ratio = 0.0
    #read the worker_config.json to init the worker node
    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    worker_num = len(workers_config['worker_config_list'])

    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    para_nums = torch.nn.utils.parameters_to_vector(global_model.parameters()).nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model Size: {} MB".format(model_size))

    # create workers
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       client_ip=worker_config['ip_address'],
                                       master_ip=SERVER_IP,
                                       master_port=common_config.master_listen_port_base+worker_idx,
                                       custom=custom),
                   common_config=common_config, 
                   user_name=worker_config['user_name'],
                   para_nums=para_nums
                   )
        )
    #到了这里，worker已经启动了

    # Create model instance

    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, args.data_pattern)

    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(common_config.worker_list, action="init")

    global_model.to(device)
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    #test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, selected_idxs=client_config.custom["test_data_idxes"], shuffle=False)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)


    total_time=0.0
    #local_steps_list=[50,40,50,30,50,40,30,50,40,30]
    #compre_ratio_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    local_steps_list=[50,50,50,50,50,50,50,50,50,50]
    compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
    computation_resource=[3,1,6,7,7,5,5,2,6,2]
    bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    #computation_resource=[9,3,5,1,4,6,4,1,4,7]
    #bandwith_resource=[8,7,7,6,9,5,3,3,5,4]
    total_resource=0.0
    total_bandwith=0.0
    #computation_resource,bandwith_resource=random_RC(10)
    RESULT_PATH = 'result//result.txt'
    result_out = open(RESULT_PATH, 'a+')

    #local_steps,compre_ratio=40,0.5
    for epoch_idx in range(1, 1+common_config.epoch):
        communication_parallel(common_config.worker_list, action="send_para", data=None)

        print("get begin")
        communication_parallel(common_config.worker_list, action="get_para")
        communication_parallel(common_config.worker_list, action="get_time")
        print("get end")

        global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
        global_para = aggregate_model_with_memory(global_para, common_config.worker_list, args.step_size)
        print("send begin")
        communication_parallel(common_config.worker_list, action="send_model",data=global_para)
        torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
        print("send end")

        test_loss, acc = test(global_model, test_loader, device, model_type=args.model_type)
        common_config.recoder.add_scalar('Accuracy/average', acc, epoch_idx)
        common_config.recoder.add_scalar('Test_loss/average', test_loss, epoch_idx)
        print("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))

        local_steps_list,compre_ratio_list,sum_time=update_E(common_config.worker_list,local_steps_list,compre_ratio_list)
        total_time=total_time+sum_time
        total_resource=total_resource+Sum(computation_resource,local_steps_list)*1.3
        total_bandwith=total_bandwith+Sum(bandwith_resource,compre_ratio_list)*1
        print(total_time,total_resource,total_bandwith)
        common_config.recoder.add_scalar('Accuracy/average_time', acc, total_time)
        common_config.recoder.add_scalar('Test_loss/average_time', test_loss, total_time)
        common_config.recoder.add_scalar('resource_time', total_resource, total_time)
        common_config.recoder.add_scalar('bandwith_time', total_bandwith, total_time)
        common_config.recoder.add_scalar('resource_epoch', total_resource, epoch_idx)
        common_config.recoder.add_scalar('bandwith_epoch', total_bandwith, epoch_idx)
        result_out.write('{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx,total_time,total_bandwith,total_resource,acc,test_loss))
        result_out.write('\n')
        #for worker in common_config.worker_list:
        #    worker.config.local_steps=local_steps_list[(worker.config.idx+epoch_idx)%10]
        #    worker.config.compre_ratio=compre_ratio_list[(worker.config.idx+epoch_idx)%10]
        #local_steps_list,compre_ratio_list=update_E_C(common_config.worker_list,local_steps_list,compre_ratio_list)
        print(local_steps_list)
        print(compre_ratio_list)
        
    # close socket
    result_out.close()
    for worker in common_config.worker_list:
        worker.socket.shutdown(2)
    
def Sum(list1,list2):
    sum=0.0
    for idx in range(0, len(list1)):
        sum=sum+float(list1[idx])*float(list2[idx])
    return sum

def random_RC(num):
    computation_resource=np.random.randint(1,num,num)
    bandwith_resourc=np.random.randint(1,num,num)
    return computation_resource,bandwith_resourc

def update_E(worker_list,local_steps_list,compre_ratio_list):
    local_steps=random.randint(40,60)
    compre_ratio=local_steps/200.0
    train_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    send_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    min_train_time=10000.0
    min_train_time_idx=1
    min_send_time=10000.0
    min_send_time_idx=1
    sum_local_steps=0
    for worker in worker_list:
        train_time_list[worker.config.idx]=worker.config.train_time
        send_time_list[worker.config.idx]=worker.config.send_time
        if train_time_list[worker.config.idx]<min_train_time:
            min_train_time=train_time_list[worker.config.idx]
            min_train_time_idx=worker.config.idx
        if send_time_list[worker.config.idx]<min_send_time:
            min_send_time=send_time_list[worker.config.idx]
            min_send_time_idx=worker.config.idx
    for worker in worker_list:
        worker.config.local_steps=int((train_time_list[min_train_time_idx]/train_time_list[worker.config.idx])*local_steps)
        worker.config.compre_ratio=(train_time_list[min_train_time_idx]/train_time_list[worker.config.idx])*compre_ratio
        #(send_time_list[min_train_time_idx]/send_time_list[worker.config.idx])*compre_ratio
        worker.config.local_steps=28
        #worker.config.local_steps=int(local_steps/2)+3
        worker.config.compre_ratio=0.4
        local_steps_list[worker.config.idx]=worker.config.local_steps
        compre_ratio_list[worker.config.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.local_steps
    for worker in worker_list:
        worker.config.average_weight=(1.0*worker.config.local_steps)/(sum_local_steps)
        
    max_train_time=max(train_time_list)
    max_send_time=max(send_time_list)
    total_time=max_train_time*30*0.4
    #local_steps/2*0.9
    #total_time=min_train_time*50+min_train_time*40
    #total_time=min_train_time*local_steps/2.0
    return local_steps_list,compre_ratio_list,total_time
'''
def update_E_C(worker_list,local_steps_list,compre_ratio_list):

    train_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    send_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    min_train_time=10000.0
    min_train_time_idx=1
    min_send_time=10000.0
    min_send_time_idx=1
    for worker in worker_list:
        train_time_list[worker.config.idx]=worker.config.train_time/local_steps_list[worker.config.idx]
        send_time_list[worker.config.idx]=worker.config.send_time/compre_ratio_list[worker.config.idx]
        if train_time_list[worker.config.idx]<min_train_time:
            min_train_time=train_time_list[worker.config.idx]
            min_train_time_idx=worker.config.idx
        if send_time_list[worker.config.idx]<min_send_time:
            min_send_time=send_time_list[worker.config.idx]
            min_send_time_idx=worker.config.idx
    for worker in worker_list:
        worker.config.local_steps=int((train_time_list[min_train_time_idx]/train_time_list[worker.config.idx])*50)
        worker.config.compre_ratio=send_time_list[min_send_time_idx]/send_time_list[worker.config.idx]
        local_steps_list[worker.config.idx]=worker.config.local_steps
        compre_ratio_list[worker.config.idx]=worker.config.compre_ratio
    return local_steps_list,compre_ratio_list

def update_E_C_rand(worker_list,local_steps_list,compre_ratio_list):
    train_time_list=np.random.randint(6,10,10)
    send_time_list=np.random.randint(6,10,10)*50
    print(train_time_list)
    print(send_time_list)
    train_time_list=train_time_list.tolist()
    send_time_list=send_time_list.tolist()
    min_train_time=min(train_time_list)
    min_train_time_idx=train_time_list.index(min_train_time)
    max_train_time=max(train_time_list)
    min_send_time=min(send_time_list)
    min_send_time_idx=send_time_list.index(min_send_time)
    max_send_time=max(send_time_list)
    sum_local_steps=0
    for worker in worker_list:
        worker.config.local_steps=int((train_time_list[min_train_time_idx]/train_time_list[worker.config.idx])*50)
        worker.config.compre_ratio=send_time_list[min_send_time_idx]/send_time_list[worker.config.idx]
        #worker.config.local_steps=10
        #worker.config.compre_ratio=1
        local_steps_list[worker.config.idx]=worker.config.local_steps
        compre_ratio_list[worker.config.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.local_steps
    for worker in worker_list:
        worker.config.average_weight=(9.0*worker.config.local_steps)/(sum_local_steps*10.0)
    #total_time=min_train_time*50+min_send_time
    total_time=max_train_time*10+max_send_time
    return local_steps_list,compre_ratio_list,total_time
'''
def aggregate_model(local_para, worker_list, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight=1.0/(len(worker_list)+1)
        for worker in worker_list:
            indice = worker.config.neighbor_indices
            #print("index of worker: ",worker.config.idx,indice)
            selected_indicator = torch.zeros_like(local_para)
            selected_indicator[indice] = 1.0
            model_delta = (worker.config.neighbor_paras - local_para) * selected_indicator
            para_delta += step_size * worker.config.average_weight * model_delta

            #client_config.estimated_consensus_distance[neighbor_idx] = np.power(np.power(torch.norm(model_delta).item(), 2) / len(indice) * model_delta.nelement(), 0.5)
        #print(para_delta)
        local_para += para_delta

    return local_para

def aggregate_model_with_memory(local_para, worker_list, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight=1.0/(len(worker_list)+1)
        for worker in worker_list:
            indice = worker.config.neighbor_indices
            #print("index of worker: ",worker.config.idx,indice)
            selected_indicator = torch.zeros_like(local_para)
            selected_indicator[indice] = 1.0
            model_delta = worker.config.neighbor_paras
            para_delta += step_size * worker.config.average_weight * model_delta

            #client_config.estimated_consensus_distance[neighbor_idx] = np.power(np.power(torch.norm(model_delta).item(), 2) / len(indice) * model_delta.nelement(), 0.5)
        #print(para_delta)
        local_para += para_delta

    return local_para

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get_para":
                tasks.append(loop.run_in_executor(executor, get_compressed_model_top,worker.config,worker.socket,worker.para_nums))
            elif action == "get_time":
                tasks.append(loop.run_in_executor(executor, get_time,worker.config,worker.socket))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
            elif action == "send_para":
                data=(worker.config.local_steps,worker.config.compre_ratio)
                tasks.append(loop.run_in_executor(executor, worker.send_data,data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def get_time(config,socket):
    train_time,send_time= get_data_socket(socket)
    config.train_time=train_time
    config.send_time=send_time
    print(config.idx," train time: ", train_time," send time: ", send_time)


def get_compressed_model_rand(config, socket, nelement):
    #start_time = time.time()
    received_para, select_n, rd_seed = get_data_socket(socket)
    received_para.to(device)
    #print(config.idx," get time: ", time.time() - start_time)

    restored_model = torch.zeros(nelement).to(device)
    
    rng = np.random.RandomState(rd_seed)
    indices = rng.choice(nelement, size=select_n, replace=False)
    restored_model[indices] = received_para

    config.neighbor_paras = restored_model.data
    config.neighbor_indices = indices

def get_compressed_model_top(config, socket, nelement):
    #start_time = time.time()
    received_para, indices = get_data_socket(socket)
    received_para.to(device)
    #print(config.idx," get time: ", time.time() - start_time)

    restored_model = torch.zeros(nelement).to(device)
    
    restored_model[indices] = received_para

    config.neighbor_paras = restored_model.data
    config.neighbor_indices = indices


def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100":
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num-data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx*worker_num:(tmp_idx+1)*worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
    elif dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if data_pattern == 0:
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            partition_sizes = [
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148,0.111],
                                [0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1472,0.112],
                                [0.1482,  0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0  , 0.111],
                                [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112, 0.0],
                                ]
        elif data_pattern == 2:
            partition_sizes = [
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    ]
        elif data_pattern == 3:
            partition_sizes = [[0.1428,  0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0],
                                [0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0],
                                [0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0],
                                [0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432],
                                [0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428],
                                ]
        elif data_pattern == 4:
            partition_sizes = [[0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                                [0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
                                [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125],
                                ]
        elif data_pattern == 5:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 6:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 7:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 8:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 9:
            non_iid_ratio = 0.9
            partition_sizes = non_iid_partition(non_iid_ratio)
        # elif data_pattern == 10:
        #     non_iid_ratio = 0.5
        #     partition_sizes = non_iid_partition(non_iid_ratio)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    # test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
    
    return train_data_partition, test_data_partition

def partition_data_old(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if data_pattern == 0:
        partition_sizes = [1.0 / worker_num for _ in range(worker_num)]
        train_data_partition = datasets.RandomPartitioner(train_dataset, partition_sizes=partition_sizes)
        test_data_partition = datasets.RandomPartitioner(test_dataset, partition_sizes=partition_sizes)
    else:
        if data_pattern == 1:
            partition_sizes = [[0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02],
                                [0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18],
                                [0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02],
                                [0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18],
                                [0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02],
                                [0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18],
                                [0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02],
                                [0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18],
                                [0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02],
                                [0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18, 0.02, 0.18],
                            ]
        elif data_pattern == 2:
            partition_sizes = [[0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                                [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2,],
                                [0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                                [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2,],
                                [0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                                [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2,],
                                [0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                                [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2,],
                                [0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                                [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2,],
                            ]
        train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
        test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)
        #test_data_partition = datasets.RandomPartitioner(test_dataset, partition_sizes=np.sum(partition_sizes, axis=0) / np.sum(partition_sizes))
    
    return train_data_partition, test_data_partition

if __name__ == "__main__":
    #print(np.random.rand(10)*10)
    print(random.randint(40,60))
    #print(np.random.randint(1,10,10))
    x = np.random.normal(loc=1, scale=np.sqrt(2),size=(1,10))
    #print(x)
    #print(np.random.normal(loc=1, scale=np.sqrt(2)))
    #print(np.random.normal(loc=1, scale=np.sqrt(2)))
    #bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    #print(bandwith_resource)
    main()
