import numpy as np
import scipy.sparse as sp
import nose
import cudamat as cm
import cudamat.sparse as spcm


def setup():
    cm.cublas_init()

def teardown():
    cm.cublas_shutdown()


def test_sparse_dot():

    m = 128
    k = 256
    n = 64
    a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='F')
    b = np.array(np.random.randn(k, n)*10, dtype=np.float32, order='F')

    alpha = 2.
    r = alpha * np.dot(a, b)

    sparse_a = sp.csr_matrix(a)

    m1 = spcm.SparseCUDAMatrix(sparse_a)
    m2 = cm.CUDAMatrix(b)
    m3 = spcm.sparse_dot(m1, m2, alpha = alpha, beta = 0)
    m3.copy_to_host()

    assert np.max(np.abs(r - m3.numpy_array)) < 10**-2, "Error in SparseCUDAMatrix.sparse_dot exceeded threshold"

if __name__ == '__main__':
    nose.runmodule()
