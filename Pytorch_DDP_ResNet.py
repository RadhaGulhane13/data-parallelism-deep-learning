import socket

# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

# import torchvision module to handle image manipulation
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import sys
import argparse
import os
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--batch-size', action='store', default=32, type=int,
    help='Batch size (default: 32)')
args = parser.parse_args()

batch_size = args.batch_size
# batch_size = 512
#print("batch_size : ", batch_size)

def get_data():
    # Download training data from open datasets.
    training_data = datasets.SVHN(
        root="data",
        split="train",
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.SVHN(
        root="data",
        split="test",
        download=True,
        transform=ToTensor(),
    )
    #Sample the batch ids based on the rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            training_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size = args.batch_size, 
                                sampler=train_sampler, num_workers= 0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                sampler=test_sampler, num_workers= 0)

    sample_size = len(train_sampler)

    return [train_dataloader, test_dataloader, sample_size]


def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default
my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
os.environ["CUDA_VISIBLE_DEVICES"]=str(my_local_rank)

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

dev = 'cuda:0'

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def run(world_rank, world_size, train_dataloader, test_dataloader, sample_size):
    local_rank = 0 
    local_world_size = 1 #For one GPU per node
    count_device = torch.cuda.device_count() 
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    # print(
    #     f"[{os.getpid()}] rank = {dist.get_rank()}, "
    #     + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    # )
    
    device = torch.device("cuda:{}".format(local_rank))
    model = torchvision.models.resnet50().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    num_epochs = 3
    
    ddp_model.train()
    for epoch in range(num_epochs):
        #print("World Rank: {}, Epoch: {}, Training ...".format(dist.get_rank(), epoch + 1))
        time_start = time.time()
        for data in train_dataloader:
            images = data[0].to(device)
            labels = data[1].to(device)            
            preds = ddp_model(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_end = time.time() - time_start
        #print('Rank: {} end_time {}'.format(dist.get_rank(), time_end))
        # if dist.get_rank() == 0:
        print('Rank: {} Epoch {}: train set accuracy {:.2f} time {:.2f} throughput {} Sample Size {}'\
        .format(dist.get_rank(), epoch + 1, evaluate(ddp_model, device, \
        test_dataloader), time_end, int(sample_size//time_end), sample_size))
            
def init_processes(fn, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend)
    size = dist.get_world_size()
    rank = dist.get_rank()
    train_dataloader, test_dataloader, sample_size = get_data()
    fn(rank, size, train_dataloader, test_dataloader, sample_size)


if __name__ == "__main__":
    init_processes(run, backend='mpi')