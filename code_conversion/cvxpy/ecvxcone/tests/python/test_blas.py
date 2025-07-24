import ctypes
import numpy as np
import time
from cvxopt import misc, matrix, blas

np.set_printoptions(suppress=True)

N = int(2e6)

def test_blas_gemv():
    print("==== Running test_blas_gemv ====")
    
    A = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    # print(f"A is: {A}")
    x = matrix(np.identity(2, dtype=np.float64), tc='d')
    y = matrix(np.zeros((3, 1), dtype=np.float64), tc='d')  # output buffer for C
    
    blas.gemv(A, x, y)
    t1 = time.time()
    for i in range(N):
        blas.gemv(A, x, y)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    
    # print(f"C is: {C}")


def test_blas_trsm():
    print("==== Running test_blas_trsm ====")
    
    A_np = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0]
    ])
    A = matrix(A_np, tc='d')

    # B 向量或矩阵（3x1）
    B_np = np.array([
        [7.0],
        [8.0],
        [9.0]
    ])
    B = matrix(B_np, tc='d')
    
    blas.trsm(A, B)
    t1 = time.time()
    for i in range(N):
        blas.trsm(A, B)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    
def test_blas_ediv():
    A = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    # print(f"A is: {A}")
    C = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    B = base.div(A, C)
    t1 = time.time()
    for i in range(N):
        B = base.div(A, C)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    
    
def test_base_sqrt():
    A = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    # print(f"A is: {A}")
    C = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    B = base.sqrt(A)
    t1 = time.time()
    for i in range(N):
        B = base.sqrt(A)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    
def test_blas_all():
    print("==== Running test_blas ====")
    
    # test_blas_gemv()
    test_blas_trsm()
    
    # Test base_emul
    # test_base_emul()  # Uncomment if you want to run this test
    
    # Test base_ediv
    # test_base_ediv()  # Uncomment if you want to run this test
    # test_base_sqrt()  # Uncomment if you want to run this test


if __name__ == "__main__":
    test_blas_all()
