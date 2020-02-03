""" Personal Utilities """

class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(self, *args, **kwargs)
        self.__dict__ = self
