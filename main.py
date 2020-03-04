#!/usr/bin/env python3
# Standard import {{{

# Third-party import
import click
import pandas as pd

# Local import

# }}}

# Golbal Settings {{{
_global_test_options = [
    click.option('-test', '--test-arg', 'var_name', default='default value', help='Please customize option value'),
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
    df = pd.DataFrame(kwargs.items(), columns=['Name', 'Value'])
    print(df.to_string(index=False))

    # do


if "__main__" == __name__:
    main()
