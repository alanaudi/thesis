#!/usr/bin/env python
# Standard import {{{
import time
# Third-party import
import click
from tabulate import tabulate
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
# Local import
from lib.utils import EasyDict
from lib.nn.modules import CNN
# }}}

# Golbal Settings {{{
_global_test_options = [
        # click.option('-test', '--test-arg', 'var_name', default='default value', help='Please customize option value'),
        click.option('-b', '--batch-size', 'batch_size', default=64, help='Input batch size for training (default: 64)'),
        click.option('-bte', '--test-batch-size', 'test_batch_size', default=1000, help='Input batch size for testing (default: 1000)'),
        click.option('-e', '--epochs', 'epochs', default=14, help='Number of epochs to train (deafult: 14)'),
        click.option('-lr', '--learning-rate', 'learning_rate', default=1.0, help='Learning rate (default: 1.0)'),
        click.option('-ga', '--gamma', 'gamma', default=0.7, help='Learning rate step gamma (default: 0.7)'),
        click.option('-s', '--seed', 'seed', default=1, help='Random seed (default: 1)'),
        click.option('-li', '--log-interval', 'log_interval', default=10, help='How many batches to wait before logging training status (deafult: 10)'),
        click.option('-m', '--save-model', 'save_model', default=False, help='Save current model (default: False)'),
        ]

def global_test_options(func):
    for option in reversed(_global_test_options):
        func = option(func)
    return func
# }}}

@click.group()
@global_test_options
def main(**kwargs): # {{{
    pass
# }}}

@main.command()
@global_test_options
@click.option('--batch-size', 'batch_size', default=4, help='Base directory')
@click.option('--num-workers', 'num_workers', default=2, help='Base directory')
@click.option('--learning-rate', 'learning_rate', default=0.001, help='Base directory')
@click.option('--momentum', 'momentum', default=0.9, help='Base directory')
def cnn(**kwargs): # {{{
    # Print argument, option, parameter
    print(tabulate(list(kwargs.items()), headers=['Name', 'Value'], tablefmt='orgtbl'))
    args = EasyDict(kwargs)

    # Load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Show data
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Neural Network
    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer =optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # Train
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished training")
# }}}
def imshow(img): # {{{
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# }}}

if "__main__" == __name__:
    main()
