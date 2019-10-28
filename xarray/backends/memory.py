import copy
from collections import OrderedDict

import numpy as np

from ..core.variable import Variable
from .common import AbstractWritableDataStore


class InMemoryDataStore(AbstractWritableDataStore):
    """Stores variables and attributes directly in OrderedDicts.

    This store exists for internal testing purposes, e.g., for integration
    tests with dask.array that will not need to write actual data to disk.
    """
    def __init__(self):
        self._variables = OrderedDict()
        self._attributes = OrderedDict()
        # do we need locks? are writes to NumPy arrays thread-safe?
        # this is a dumb but safe approach.
        self._write_locks = defaultdict(threading.Lock)

    def get_variables(self):
        return self._variables

    def get_attributes(self):
        return self._attributes

    def get_read_lock(self, name, region=Ellipsis):
        return None

    def create_variable(self, name, variable, check_encoding=False):
        if check_encoding and variable.encoding:
            raise ValueError('encoding must be empty')
        store_variable = Variable(variable.dims,
                                  np.empty_like(variable),
                                  copy.deepcopy(variable.attrs))
        self._variables[name] = store_variable
        return store_variable.values

    def get_writable_array(self, name):
        return self._variables[name].values

    def set_attribute(self, name, value):
        self._attributes[name] = copy.deepcopy(value)

    def get_write_lock(self, name, region=Ellipsis):
        return self._write_locks[name]
