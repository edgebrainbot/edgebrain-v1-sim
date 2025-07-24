from cvxopt import matrix, misc
import numpy as np
import time

s_input = "../c/input_matrix/s.npy"
z_input = "../c/input_matrix/z.npy"
lmbda_input = "../c/input_matrix/lmbda.npy"

N = int(1e3)  # Number of iterations for performance testing
np.set_printoptions(suppress=True)

def test_misc_unpack():
    from math import sqrt

    # Define cone dimensions: 1 linear, 1 SOC, 1 SDP (2x2)
    dims = {'l': 1, 'q': [1], 's': [2]}

    # Construct packed x: [l=1, q=2, SDP=[s11=3, s12=4*sqrt(2), s22=5]]
    x = matrix([1.0, 2.0, 3.0, 4.0 * sqrt(2), 5.0])

    y = matrix(0.0, (6, 1))  # Unpacked matrix of size 6x1
    # Unpack into full matrix format
    misc.unpack(x, y, dims)

    print("Unpacked y:")
    for i in range(len(y)):
        print(f"y[{i}] = {y[i]:.6f}")

    # Expected full matrix in column-major order:
    # [s11, s21, s12, s22] = [3.0, 4.0, 4.0, 5.0]
    expected = [1.0, 2.0, 3.0, 4.0, 4.0, 5.0]

    # Compare
    assert np.allclose(y, expected, atol=1e-6), "Test FAILED"
    print("Test PASSED")


def test_misc_symm():
    # 初始化下三角矩阵（列主序）
    # 目标：对称化填充上三角
    # C 中的顺序是 column-major: [A(0,0), A(1,0), A(2,0), A(0,1), A(1,1), A(2,1), A(0,2), A(1,2), A(2,2)]
    buf = [
        1.0, 2.0, 4.0,  # column 0
        0.0, 3.0, 5.0,  # column 1 （上三角 0.0 是 placeholder，将被填）
        0.0, 0.0, 6.0   # column 2
    ]
    A = matrix(buf, (3, 3), 'd')  # 列主序

    # 调用 cvxopt.misc.symm：将下三角填充到上三角
    misc.symm(A, 3)
    
    t1 = time.time()
    # 执行 SVD: A = U * diag(s) * Vt
    for i in range(N):
        misc.symm(A, 3)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")

    # 打印对称矩阵
    print("Symmetric matrix A after misc.symm:")
    for i in range(3):
        for j in range(3):
            print(f"{A[i,j]:8.4f}", end=' ')
        print()


def test_misc_scaling():
    s = np.load(s_input)
    z = np.load(z_input)
    lmbda = np.load(lmbda_input)

    dims = {
        'l': 0,
        'q': [],
        's': [10, 6, 20, 16, 10, 3, 6, 6],
    }

    s = matrix(s)
    z = matrix(z)
    lmbda = matrix(lmbda)

    W = misc.compute_scaling(s, z, lmbda, dims, None)
    # misc.update_scaling(W, lmbda, s, z)
    
    t1 = time.time()
    for i in range(N):
        W = misc.compute_scaling(s, z, lmbda, dims, None)
        # misc.update_scaling(W, lmbda, s, z)
    t2 = time.time()
    print(f"Compute Scaling Time taken: {t2 - t1:.6f} seconds")
    
    t1 = time.time()
    for i in range(N):
        # W = misc.compute_scaling(s, z, lmbda, dims, None)
        misc.update_scaling(W, lmbda, s, z)
    t2 = time.time()
    print(f"Updating Scaling Time taken: {t2 - t1:.6f} seconds")
    print(f"Computed scaling d: {W['d']}")

def test_misc_all():
    print("Running all CVXOPT misc tests...")

    # Test unpacking
    # test_cvxopt_misc_unpack()

    # Test symmetric matrix filling
    test_misc_symm()
    
    test_misc_scaling()

    print("All CVXOPT misc tests completed successfully!")


if __name__ == "__main__":
    test_misc_all()