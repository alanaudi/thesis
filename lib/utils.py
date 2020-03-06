""" Personal Utilities """
# Standard import {{{
import csv

# Third-party import
import xlrd

# Local import

# }}}


class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def xlsx2csv(filename):
    wb = xlrd.open_workbook(filename)
    for name in wb.sheet_names():
        sh = wb.sheet_by_name(name)
        with open(F"{name}.csv", "w") as f:
            wr = csv.writer(f)

            for row in range(sh.nrows):
                wr.writerow(sh.row_values(row))
