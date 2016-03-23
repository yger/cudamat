
import scipy.sparse as sp
import cudamat
import ctypes as ct

import os
import platform
import warnings
import sysconfig

import ctypes as ct
import numpy as np
def load_library(basename):
    if platform.system() == 'Windows':
       ext = '.dll'
    else:
       ext = sysconfig.get_config_var('SO')
    return ct.cdll.LoadLibrary(os.path.join(
        os.path.dirname(__file__) or os.path.curdir,
        basename + ext))


_cudamat = load_library('libcudamat')

_cudamat.sparse_dot.restype = ct.c_int

class sparse_data(ct.Structure):
    _fields_ = [('indptr', ct.POINTER(ct.c_int)),
                ('indices', ct.POINTER(ct.c_int)),
                ('data', ct.POINTER(ct.c_float))]

class cudamat_sparse(ct.Structure):
    _fields_ = [('data_host', sparse_data),
                ('data_device', sparse_data),
                ('on_device', ct.c_int),
                ('on_host', ct.c_int),
                ('size', ct.c_int * 2),
                ('is_trans', ct.c_int),
                ('owns_data', ct.c_int),
                ('nnz', ct.c_int)]

class SparseCUDAMatrix(object):
    """ A SparseCUDAMatrix object represents a scipy.sparse.csr matrix of single
    precision floats on a GPU.
    """
    def __init__(self, array, copy_to_device = True, copy_on_host=True):
        """
        Initializes a new sparse matrix object on the GPU. array needs to be 
        a scipy CSR matrix. If the copy_to_device flag is set to True, 
        the GPU matrix is initialized with the given scipy array. If array is not 
        a CSR scipy matrix, but a dense matrix, the user should consider using
        a CUDAMatrix. If the copy_on_host flag is set to True, a copy of the matrix 
        will be created in host memory even if the matrix is of the correct type 
        (float32).
        """
        assert(type(array) == sp.csr_matrix)

        array.data = cudamat.reformat(array.data, copy=copy_on_host)

        self.mat = cudamat_sparse()
        self.size = self.mat.size
        self.p_mat = ct.pointer(self.mat)
        self.scipy_array = array

        _cudamat.init_from_sparse_array(self.p_mat,
                                        self.scipy_array.data.ctypes.data_as(ct.POINTER(ct.c_float)),
                                        self.scipy_array.indices.ctypes.data_as(ct.POINTER(ct.c_int)),
                                        self.scipy_array.indptr.ctypes.data_as(ct.POINTER(ct.c_int)),
                                        ct.c_int(self.scipy_array.shape[0]), ct.c_int(self.scipy_array.shape[1]),
                                        ct.c_int(self.scipy_array.nnz))
        if copy_to_device:
          err_code = _cudamat.copy_sparse_to_device(self.p_mat)
          if err_code:
            raise generate_exception(err_code)

        # Keep a reference to free device memory in case of a crash.
        self.__free_device_memory = _cudamat.free_device_memory

    @property
    def shape(self):
        return self.mat.size[0], self.mat.size[1]


def sparse_dot(sparse_mat, dense_mat, beta=0., alpha=1.0, target = None):
    """
    Find the dot product between a sparse matrix m1 and dense matrix matrix m2 
    and store in target:
    target = beta*target + alpha*(m1 m2)
    If no target is given, it will be created automatically, but not
    initialized -- so beta should be left at its default value zero.
    """
    if not target:
        m = sparse_mat.shape[0]
        n = dense_mat.shape[1]
        target = cudamat.empty((m, n))

    err_code = _cudamat.sparse_dot(sparse_mat.p_mat, dense_mat.p_mat, target.p_mat, ct.c_float(beta), ct.c_float(alpha))
    if err_code:
        raise generate_exception(err_code)

    return target