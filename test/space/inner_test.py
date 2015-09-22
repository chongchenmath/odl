# Copyright 2014, 2015 The ODL development group
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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import range

# External module imports
import unittest
import numpy as np
import scipy as sp

# ODL
import odl
from odl.util.testutils import ODLTestCase


class MatrixWeightedInnerTest(ODLTestCase):
    @staticmethod
    def _vectors(fn):
        # Generate numpy vectors, real or complex
        if isinstance(fn, odl.Rn):
            xarr = np.random.rand(fn.size)
            yarr = np.random.rand(fn.size)
        else:
            xarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)
            yarr = np.random.rand(fn.size) + 1j * np.random.rand(fn.size)

        # Make Fn vectors
        x = fn.element(xarr)
        y = fn.element(yarr)
        return xarr, yarr, x, y

    @staticmethod
    def _sparse_matrix(fn):
        nnz = np.random.randint(0, fn.size**2/2)
        coo_r = np.random.randint(0, fn.size, size=nnz)
        coo_c = np.random.randint(0, fn.size, size=nnz)
        values = np.random.rand(nnz)
        return sp.sparse.coo_matrix((values, (coo_r, coo_c)),
                                    shape=(fn.size, fn.size))

    @staticmethod
    def _dense_matrix(fn):
        return np.asmatrix(np.random.rand(fn.size, fn.size), dtype=float)

    def test_init(self):
        rn = odl.Rn(10)
        sparse_mat = self._sparse_matrix(rn)
        dense_mat = self._dense_matrix(rn)

        # Just test if the code runs
        inner = odl.MatrixWeightedInner(sparse_mat)
        inner = odl.MatrixWeightedInner(dense_mat)

        nonsquare_mat = np.eye(10, 5)
        with self.assertRaises(ValueError):
            odl.MatrixWeightedInner(nonsquare_mat)

    def _test_equals(self, n):
        rn = odl.Rn(n)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)

        inner_sparse = odl.MatrixWeightedInner(sparse_mat)
        inner_sparse2 = odl.MatrixWeightedInner(sparse_mat)
        inner_sparse_as_dense = odl.MatrixWeightedInner(sparse_mat_as_dense)
        inner_dense = odl.MatrixWeightedInner(dense_mat)
        inner_dense2 = odl.MatrixWeightedInner(dense_mat)

        self.assertEquals(inner_sparse, inner_sparse)
        self.assertEquals(inner_sparse, inner_sparse2)
        self.assertEquals(inner_sparse, inner_sparse_as_dense)
        self.assertEquals(inner_sparse_as_dense, inner_sparse)
        self.assertEquals(inner_dense, inner_dense2)

    def test_equals(self):
        for _ in range(20):
            self._test_equals(10)

    def _test_call_real(self, n):
        rn = odl.Rn(n)
        xarr, yarr, x, y = self._vectors(rn)
        sparse_mat = self._sparse_matrix(rn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(rn)

        inner_sparse = odl.MatrixWeightedInner(sparse_mat)
        inner_dense = odl.MatrixWeightedInner(dense_mat)

        result_sparse = inner_sparse(xarr, yarr)
        result_dense = inner_dense(xarr, yarr)

        true_result_sparse = np.dot(
            yarr, np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
        true_result_dense = np.dot(
            yarr, np.asarray(np.dot(dense_mat, xarr)).squeeze())

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def _test_call_complex(self, n):
        cn = odl.Cn(n)
        xarr, yarr, x, y = self._vectors(cn)
        sparse_mat = self._sparse_matrix(cn)
        sparse_mat_as_dense = sparse_mat.todense()
        dense_mat = self._dense_matrix(cn)

        inner_sparse = odl.MatrixWeightedInner(sparse_mat)
        inner_dense = odl.MatrixWeightedInner(dense_mat)

        result_sparse = inner_sparse(xarr, yarr)
        result_dense = inner_dense(xarr, yarr)

        true_result_sparse = np.dot(
            yarr.conj(),
            np.asarray(np.dot(sparse_mat_as_dense, xarr)).squeeze())
        true_result_dense = np.dot(
            yarr.conj(), np.asarray(np.dot(dense_mat, xarr)).squeeze())

        self.assertAlmostEquals(result_sparse, true_result_sparse)
        self.assertAlmostEquals(result_dense, true_result_dense)

    def test_call(self):
        for _ in range(20):
            self._test_call_real(10)
            self._test_call_complex(10)

    def test_repr(self):
        n = 10
        sparse_mat = sp.sparse.dia_matrix((np.arange(n, dtype=float), [0]),
                                          shape=(n, n))
        dense_mat = sparse_mat.todense()

        inner_sparse = odl.MatrixWeightedInner(sparse_mat)
        inner_dense = odl.MatrixWeightedInner(dense_mat)

        repr_str = 'MatrixWeightedInner()'
        self.assertEquals(repr(inner_const), repr_str)

    def test_str(self):
        constant = 1.5
        inner_const = odl.CudaConstantWeightedInner(constant)

        print_str = '(x, y) --> 1.5 * y^H x'
        self.assertEquals(str(inner_const), print_str)

if __name__ == '__main__':
    unittest.main(exit=False)
