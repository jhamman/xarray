import logging
import time
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from ..conventions import cf_encoder
from ..core import indexing
from ..core.pycompat import dask_array_type
from ..core.utils import FrozenOrderedDict, NdimSizeLenMixin
from .locks import combine_locks

# Create a logger object, but don't add any handlers. Leave that to user code.
logger = logging.getLogger(__name__)


NONE_VAR_NAME = '__values__'


def _encode_variable_name(name):
    if name is None:
        name = NONE_VAR_NAME
    return name


def _decode_variable_name(name):
    if name == NONE_VAR_NAME:
        name = None
    return name


def find_root_and_group(ds):
    """Find the root and group name of a netCDF4/h5netcdf dataset."""
    hierarchy = ()
    while ds.parent is not None:
        hierarchy = (ds.name,) + hierarchy
        ds = ds.parent
    group = '/' + '/'.join(hierarchy)
    return ds, group


def robust_getitem(array, key, catch=Exception, max_retries=6,
                   initial_delay=500):
    """
    Robustly index an array, using retry logic with exponential backoff if any
    of the errors ``catch`` are raised. The initial_delay is measured in ms.

    With the default settings, the maximum delay will be in the range of 32-64
    seconds.
    """
    assert max_retries >= 0
    for n in range(max_retries + 1):
        try:
            return array[key]
        except catch:
            if n == max_retries:
                raise
            base_delay = initial_delay * 2 ** n
            next_delay = base_delay + np.random.randint(base_delay)
            msg = ('getitem failed, waiting %s ms before trying again '
                   '(%s tries remaining). Full traceback: %s' %
                   (next_delay, max_retries - n, traceback.format_exc()))
            logger.debug(msg)
            time.sleep(1e-3 * next_delay)


class BackendArray(NdimSizeLenMixin, indexing.ExplicitlyIndexed):

    def __array__(self, dtype=None):
        key = indexing.BasicIndexer((slice(None),) * self.ndim)
        return np.asarray(self[key], dtype=dtype)


class ArrayWriter(object):
    def __init__(self):
        self.sources = []
        self.targets = []
        self.locks = []

    def add(self, source, target, lock=None):
        if isinstance(source, dask_array_type):
            self.sources.append(source)
            self.targets.append(target)
            self.locks.append(lock)
        else:
            target[...] = source

    def sync(self):
        if self.sources:
            import dask.array as da
            # TODO: dask.array.store needs to be able to accept a list of Lock
            # objects. For now, using the CombinedLock approach
            da.store(self.sources, self.targets,
                     lock=combine_locks(self.locks))


class AbstractDataStore(Mapping):
    """An abstract interface for implementing datastores.

    Every method is something that should potentially be implemented by
    developers of new datastores.
    """

    def get_variables(self):
        # type: (Any,) -> Mapping[Any, xarray.Variable]
        """Return a map from names to xarray.Variable objects.

        Consider returning Variables whose data is non-eagerly evaluated, e.g.,
        by wrapping with xarray.core.utils.LazilyIndexedArray or
        using dask.array.

        # TODO: move xarray.core.utils.LazilyIndexedArray to public API.
        """
        raise NotImplementedError

    def get_attributes(self):
        # type: () -> Mapping
        """Return a map of global attributes on the DataStore."""
        raise NotImplementedError

    def close(self):
        """Close any resources associated with this DataStore."""
        pass

    def get_read_lock(self, name, region=Ellipsis):
        # type: (Hashable, Union[Ellipsis, Tuple[slice, ...]]) -> object
        """Return a lock for reading a region of a given variable.

        This method may be useful for DataStores that from which data is read in
        parallel (e.g., with dask).

        Parameters
        ----------
        name : Hashable
            Variable name.
        region : Union[Ellipsis, Tuple[slice, ...]], optional
            Region in the variable, e.g., valid key argument to `data[key]`,
            for which to get a lock.

        Returns
        -------
        threading.Lock() ducktype (that is, an object with `acquire` and
        `release` methods), or None, if no lock is necessary.
        """
        # Do we actually want the region argument?
        # For a library such as HDF5, this will simply return a global lock for
        # all files.
        return None

    def get_token(self):
        """Return a token identifier suitable for use by dask."""
        return str(uuid.uuid4())

    def get_name(self):
        """Return a user-friendly name for prefixing names of dask arrays.

        Not required to be unique.
        """
        return ''


class OnDiskMixin(object):

    def get_token(self):
        return (self.filename, os.path.getmtime(self.filename))

    def get_name(self):
        return 'read-from-disk'


class AbstractWritableDataStore(AbstractDataStore):
    """An abstract interface for implementing writable datastores."""

    def create_variable(self, name, variable, check_encoding=False):
        # type: (Hashable, xarray.Variable) -> WritableDuckArray
        """Create a new variable for writing into the DataStore.

        This method is responsible for setting up a variable to write. It
        SHOULD NOT actually write array values, but rather create an array to
        which xarray itself will write.

        If the given variable cannot be stored on the DataStore, this method
        MUST raise an error.

        Parameters
        ----------
        name : Hashable
            Variable name. If a variable with this name already exists in the
            DataStore, this method MAY raise an error.
        variable : xarray.Variable
            Variable to copy into the DataStore. `variable.encodings` provides
            a dictionary of DataStore specific options for how to save
            variables.
        check_encoding : bool, optional
            If True, this method SHOULD raise an error for any unexpected keys
            or invalid values in `variable.encoding`.

        Returns
        -------
        Array-like object that writes data to the store when assigning a NumPy
        array to a tuple of slice objects, e.g., ``x[key] = value``, where
        ``key`` has type ``Tuple[slice, ...]`` and length equal to the
        dimensionality of the array, and ``value`` is a ``numpy.ndarray``.
        """
        raise NotImplementedError

    def get_writable_array(self, name):
        # type: (Hashable,) -> WritableDuckArray
        """Return a writable array corresponding to an existing variable.

        This method is only needed if you want the DataStore to support partial
        writes, e.g., appending to existing variables.

        Parameters
        ----------
        name : Hashable
            Variable name. How to handle non-existing names is up to the
            DataStore class. However, xarray will never call this method unless
            a variable with the given name has already been verified to exist
            as a member of the mapping returned by `get_variables()`.

        Returns
        -------
        Writable array-like object, see `create_variable` for details.
        """
        # Note: this mostly exists for the benefit of future support for partial
        # reads -- we don't actually make use of this in the current version of
        # xarray.
        raise NotImplementedError

    def set_attribute(self, name, value):
        # type: (Hashable, Any) -> None
        """Set a global attribute on the DataStore."""
        raise NotImplementedError

    def sync(self):
        """Synchronize writes to this DataStore."""
        pass

    def get_write_lock(self, name, region=Ellipsis):
        # type: (Hashable, Union[Ellipsis, Tuple[slice, ...]]) -> object
        """Return a lock for writing a given variable.

        This method may be useful for DataStores that from which data is
        written in parallel (e.g., with dask).

        Parameters
        ----------
        name : Hashable
            Variable name.
        region : Union[Ellipsis, Tuple[slice, ...]], optional
            Region in the variable, e.g., valid key argument to `data[key]`,
            for which to get a lock.

        Returns
        -------
        threading.Lock() ducktype (that is, an object with `acquire` and
        `release` methods), or None, if no lock is necessary.
        """
        # Again, we actually have a use for the region argument? Could be useful
        # to ensure writes to zarr are safe.
        return None
