# -*- coding: utf-8 -*-
# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import os
#Modules for Distributed Training
import horovod.torch as hvd
from filelock import FileLock

parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--batch-size', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

parser.add_argument(
    '--batch-size', action='store', default=32, type=int,
    help='Batch size (default: 32)')

args = parser.parse_args()

# Use standard SVHN dataset

hvd.init()

torch.cuda.set_device(hvd.local_rank())

"""
The following library call downloads the training set and puts it into data/, 
and prepares the dataset to be passed into a pytorch as a tensor.
"""
with FileLock(os.path.expanduser("~/.horovod_lock")):
  # Download training data from open datasets.
  train_set = torchvision.datasets.SVHN(
      root="data",
      split="train",
      download=True,
      transform = transforms.Compose([
          transforms.ToTensor()                                 
      ])
  )

  # Download test data from open datasets.
  test_set = torchvision.datasets.SVHN(
      root="data",
      split="test",
      download=True,
      transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
  )

#Sample the batch ids based on the rank
train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=hvd.size(), rank=hvd.rank())
test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=hvd.size(), rank=hvd.rank())

def get_accuracy(model,dataloader):
  count=0
  correct=0

  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      images = batch[0].cuda()
      labels = batch[1].cuda()
      preds=network(images)
      batch_correct=preds.argmax(dim=1).eq(labels).sum().item()
      batch_count=len(batch[0])
      count+=batch_count
      correct+=batch_correct
  model.train()
  return correct/count

lr=0.001
shuffle=True
epochs=3

network = torchvision.models.resnet50()
network.cuda()

loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, 
                                     sampler=train_sampler, num_workers= 0)
optimizer = optim.Adam(network.parameters(), lr=lr)
size = len(loader.dataset)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(network.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=network.named_parameters(),
                                     op=hvd.Average)

print('Batch Size {}, Sample Size {}'.format(args.batch_size, size))
# set the network to training mode
network.train()
for epoch in range(epochs):
  time_start = time.time()
  for batch in loader:
    images = batch[0].cuda()
    labels = batch[1].cuda()
    preds = network(images)
    loss = F.cross_entropy(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  time_end = time.time() - time_start
  #if(hvd.rank()==0):
  print('Rank : {} Epoch {}: train set accuracy {:.2f} time {:.2f} throughput {} Sample Size {}'\
  .format(hvd.rank(), epoch + 1, get_accuracy(network,loader), 
  time_end, int(len(train_sampler)//time_end), len(train_sampler)))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size)
# if(hvd.rank()==0):
#   print('Epoch {0}: test set accuracy {1}'.format(
#         epoch,get_accuracy(network,test_loader)))


