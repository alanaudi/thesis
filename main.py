#!/usr/bin/env python3
# Standard import {{{

# Third-party import
import click
from tabulate import tabulate
import pandas as pd

# Local import
from lib.utils import EasyDict
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
def mnist(**kwargs):
    # Print argument, option, parameter
    print(tabulate(list(kwargs.items()), headers=['Name', 'Value'], tablefmt='orgtbl'))

    # do
    args = EasyDict(kwargs)
    print(args.out_dir)
if "__main__" == __name__:
    main()
