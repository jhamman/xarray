import numpy as np

from xarray import Dataset, DataArray
from xarray.core import ops
from xarray.core.variable import as_variable


def _decode_string_data(data):
    if data.dtype.kind == 'S':
        return np.core.defchararray.decode(data, 'utf-8', 'replace')
    return data


def _data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    if any(arr.dtype.kind == 'S' for arr in [arr1, arr2]):
        arr1 = _decode_string_data(arr1)
        arr2 = _decode_string_data(arr2)
    exact_dtypes = ['M', 'm', 'O', 'U']
    if any(arr.dtype.kind in exact_dtypes for arr in [arr1, arr2]):
        return ops.array_equiv(arr1, arr2)
    else:
        return ops.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)


def _assert_variable_equal(v1, v2):
    assert as_variable(v1).equals(v2), (v1, v2)


def _assert_variable_identical(v1, v2):
    assert as_variable(v1).identical(v2), (v1, v2)


def _assert_variable_all_close(v1, v2, rtol=1e-05, atol=1e-08):
    _assert_equal(v1.dims, v2.dims)
    allclose = _data_allclose_or_equiv(
        v1.values, v2.values, rtol=rtol, atol=atol)
    assert allclose


def _assert_variable_not_equal(v1, v2):
    assert as_variable(v1).equals(v2)


# TODO: write a generic "assert_equal" that uses the equals method
def _assert_equal(a1, a2):
    assert a1 == a2 or (a1 != a1 and a2 != a2)


def build_err_msg(objs, err_msg, verbose=True, precision=8):

    if isinstance(objs[0], Dataset):
        return ''
    elif isinstance(objs[0], DataArray):
        return ''


def assert_dataset_equal(actual, expected, err_msg=None, verbose=True):
    """
    Raises an AssertionError if two datasets are not equal

    Parameters
    ----------
    actual : array_like
        The object to check.
    expected : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and expected are not equal up to specified precision.

    """
    equal = actual.equals(expected)

    if equal:
        return

    if err_msg is None:
        err_msg = build_err_msg((actual, expected), verbose=verbose)

    raise AssertionError(err_msg)


def assert_dataset_identical(actual, expected, err_msg=None, verbose=True):
    equal = actual.identical(expected)

    if equal:
        return

    if err_msg is None:
        err_msg = build_err_msg((actual, expected), verbose=verbose)

    raise AssertionError(err_msg)


def assert_dataset_all_close(actual, expected, rtol=1e-05, atol=1e-08):

    _assert_equal(sorted(actual, key=str), sorted(expected, key=str))
    # TODO: assert_items_equal(actual.coords, expected.coords)
    for k in actual:
        v1 = actual.variables[k]
        v2 = expected.variables[k]
        _assert_variable_all_close(v1, v2, rtol=rtol, atol=atol)


def assert_coordinates_equal(actual, expected):
    _assert_equal(sorted(actual.coords), sorted(expected.coords))
    for k in actual.coords:
        v1 = actual.coords[k]
        v2 = expected.coords[k]
        _assert_variable_equal(v1, v2)


def assert_data_array_equal(ar1, ar2):
    _assert_variable_equal(ar1, ar2)
    assert_coordinates_equal(ar1, ar2)


def assert_data_array_identical(ar1, ar2):
    _assert_equal(ar1.name, ar2.name)
    assert_dataset_identical(ar1._to_temp_dataset(), ar2._to_temp_dataset())


def assert_data_array_all_close(ar1, ar2, rtol=1e-05, atol=1e-08):
    _assert_variable_all_close(ar1, ar2, rtol=rtol, atol=atol)
    assert_coordinates_equal(ar1, ar2)
