﻿# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Utility functions for space implementations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.space.entry_points import TENSOR_SET_IMPLS, TENSOR_SPACE_IMPLS
from odl.util import is_scalar_dtype


__all__ = ('vector', 'tensor_set', 'tensor_space', 'cn', 'rn')


def vector(array, dtype=None, impl='numpy'):
    """Create an n-tuples type vector from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
    dtype : optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    impl : string
        The backend to use. See `odl.space.entry_points.TENSOR_SET_IMPLS` and
        `odl.space.entry_points.TENSOR_SPACE_IMPLS` for available options.

    Returns
    -------
    vec : `GeneralizedTensor`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    >>> vector([1, 2, 3])  # No automatic cast to float
    tensor_space(3, 'int').element([1, 2, 3])
    >>> vector([1, 2, 3], dtype=float)
    rn(3).element([1.0, 2.0, 3.0])
    >>> vector([1 + 1j, 2, 3 - 2j])
    cn(3).element([(1+1j), (2+0j), (3-2j)])

    Non-scalar types are also supported:

    >>> vector([True, False])
    tensor_set(2, 'bool').element([True, False])

    Scalars become a one-element vector:

    >>> vector(0.0)
    rn(1).element([0.0])
    """
    # Sanitize input
    arr = np.array(array, copy=False, ndmin=1)

    # Validate input
    if arr.ndim > 1:
        raise ValueError('array has {} dimensions, expected 1'
                         ''.format(arr.ndim))

    # Set dtype
    if dtype is not None:
        space_dtype = dtype
    else:
        space_dtype = arr.dtype

    # Select implementation
    if space_dtype is None or is_scalar_dtype(space_dtype):
        space_constr = tensor_space
    else:
        space_constr = tensor_set

    return space_constr(len(arr), dtype=space_dtype, impl=impl).element(arr)


def tensor_set(shape, dtype, order='C', impl='numpy', **kwargs):
    """Return a tensor set with arbitrary data type.

    Parameters
    ----------
    shape : sequence of positive int
        Number of entries per axis for each element.
    dtype :
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        The backend to use. See `TENSOR_SET_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the set constructor.

    Returns
    -------
    tset : `TensorSet`

    Examples
    --------
    Set of 2x3 tensors with unsigned integer entries:

    >>> odl.tensor_set((2, 3), dtype='uint64')
    tensor_set((2, 3), 'uint64')

    One-dimensional spaces have special constructors:

    >>> odl.tensor_set((3,), dtype='uint64')
    tesnor_set(3, 'uint64')

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    """
    return TENSOR_SET_IMPLS[impl](shape, dtype, order, **kwargs)


def tensor_space(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        The backend to use. See `TENSOR_SPACE_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    tspace : `TensorSpace`

    Examples
    --------
    Space of 2x3 tensors with ``int64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space((2, 3), dtype='int64')
    tensor_space((2, 3), 'int')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    One-dimensional spaces have special constructors:

    >>> odl.tensor_space((3,), dtype='int64')
    fn(3, 'int')

    See also
    --------
    tensor_set : Set of tensors with arbitrary data type.
    """
    tspace_constr = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = tspace_constr.default_dtype()

    return tspace_constr(shape, dtype, **kwargs)


def cn(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only complex floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(ComplexNumbers())``.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        The backend to use. See `TENSOR_SPACE_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    complex_tspace : `TensorSpace`

    Examples
    --------
    Space of complex 2x3 tensors with ``complex64`` entries:

    >>> odl.cn((2, 3), dtype='complex64')
    cn((2, 3), 'complex64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'complex128'``:

    >>> ts = odl.cn((2, 3))
    >>> ts
    cn((2, 3))
    >>> ts.dtype
    dtype('complex128')

    One-dimensional spaces have special constructors:

    >>> odl.cn((3,))
    cn(3)

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    rn : Real tensor space.
    """
    cn_constr = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = cn_constr.default_dtype(ComplexNumbers())

    cn = cn_constr(shape, dtype, **kwargs)
    if not cn.is_complex_space:
        raise ValueError('data type {!r} not a complex floating-point type'
                         ''.format(dtype))
    return cn


def rn(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only real floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(RealNumbers())``.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        The backend to use. See `TENSOR_SPACE_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    real_tspace : `TensorSpace`

    Examples
    --------
    Space of real 2x3 tensors with ``float32`` entries:

    >>> odl.rn((2, 3), dtype='float32')
    rn((2, 3), 'float32')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.rn((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    One-dimensional spaces have special constructors:

    >>> odl.rn((3,))
    rn(3)

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    cn : Complex tensor space.
    """
    rn_constr = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = rn_constr.default_dtype(RealNumbers())

    rn = rn_constr(shape, dtype, **kwargs)
    if not rn.is_real_space:
        raise ValueError('data type {!r} not a real floating-point type'
                         ''.format(dtype))
    return rn


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
