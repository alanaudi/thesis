#!/usr/bin/env python3
# Standard import {{{

# Third-party import
import click
from tabulate import tabulate
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Local import
from lib.utils import EasyDict
from lib.nn.modules import Net
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
def main(**kwargs):
    pass

@main.command()
@global_test_options
@click.option('-bd', '--base-directory', default='/home/username/', help='Base directory')
def exp1(**kwargs):
    # Print argument, option, parameter
    print(tabulate(list(kwargs.items()), headers=['Name', 'Value'], tablefmt='orgtbl'))

    # do


@main.command()
@global_test_options
@click.option('-o', '--out-dir', 'out_dir', default='./log/mnist', help='Output directory')
@click.option('-d', '--data-dir', 'data_dir', default='./data', help='Data directory')
def mnist(**kwargs):
    # Print argument, option, parameter
    print(tabulate(list(kwargs.items()), headers=['Name', 'Value'], tablefmt='orgtbl'))

    # do
    args = EasyDict(kwargs)
    print(args.out_dir)
    torch.manual_seed(args.seed)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), F"{args.output_dir}/mnist_cnn.pt")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if "__main__" == __name__:
    main()
