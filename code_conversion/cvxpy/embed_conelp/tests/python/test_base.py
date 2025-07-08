import ctypes
import numpy as np
import time
from cvxopt import misc, matrix, base

np.set_printoptions(suppress=True)

N = int(2e6)

def test_base_syrk():
    A = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    # print(f"A is: {A}")
    C = matrix(np.identity(3, dtype=np.float64), tc='d')
    
    B = base.syrk(A, C, alpha=2.0, beta=1.0)
    t1 = time.time()
    for i in range(N):
        B = base.syrk(A, C, alpha=2.0, beta=1.0)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    
    # print(f"C is: {C}")


def test_base_emul():
    A = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    # print(f"A is: {A}")
    C = matrix(np.array([[1,2],[3,4],[5,6]], order='F'), tc='d')
    
    t1 = time.time()
    for i in range(N):
        B = base.mul(A, C)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    
    # print(f"C is: {C}")
    
def test_base_ediv():
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
    
def test_base_all():
    print("==== Running test_base ====")
    
    # Test base_syrk
    # test_base_syrk()
    
    # Test base_emul
    # test_base_emul()  # Uncomment if you want to run this test
    
    # Test base_ediv
    # test_base_ediv()  # Uncomment if you want to run this test
    test_base_sqrt()  # Uncomment if you want to run this test


if __name__ == "__main__":
    test_base_all()
