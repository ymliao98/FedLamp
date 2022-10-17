import os
import time
import math
import numpy as np
import random
import re

import torch
import torch.nn.functional as F

# <--Tool functions
font1 = {'color':  'black',
        'weight': 'normal',
        'size': 16,
}

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def killport(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(
        port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)

def count_dataset(loader):
    counts = np.zeros((10,))
    for _, target in loader:
        labels = target.view(-1).numpy()
        for label in labels:
            counts[label] += 1
    print("class counts:", counts)
    print("total data count:", np.sum(counts))

def printer(content, fid):
    print(content)
    content = content.rstrip('\n') + '\n'
    fid.write(content)
    fid.flush()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:>3}m {:2.0f}s'.format(m, s)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def adjust_learning_rate(args, optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = np.max(0.98 * lr, args.lr * 0.01)
        param_group['lr'] = lr

def test(model, device, data_loader, model_type=None):
    model.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            if model_type == 'LR':
                data = data.squeeze(1).view(-1, 28 * 28)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    # TODO: Record

    return test_loss, test_accuracy
