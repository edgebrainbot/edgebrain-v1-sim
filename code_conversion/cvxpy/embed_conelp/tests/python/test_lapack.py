import ctypes
import numpy as np
import time
from cvxopt import misc, matrix, lapack
from scipy.linalg import qr

np.set_printoptions(suppress=True)

N = int(5)

def test_lapack_gesvd():
    print("==== Running test_lapack_gesvd ====")
    
    # 构造 A (3x2)，列主序
    A_np = np.array([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]
    ])  # shape (3,2)
    A = matrix(A_np, tc='d')  # column-major

    # print("Original A:")
    # print(np.array(A).reshape(3,2))

    # A 会被 lapack.gesvd 就地修改
    # 创建 U (3x3), Vt (2x2), s (min(m,n)=2)
    U = matrix(0.0, (3, 3))
    Vt = matrix(0.0, (2, 2))
    s = matrix(0.0, (2, 1))  # 奇异值

    lapack.gesvd(A, s, U = U, Vt = Vt, jobu='A', jobvt='A')

    t1 = time.time()
    # 执行 SVD: A = U * diag(s) * Vt
    for i in range(N):
        lapack.gesvd(A, s, U = U, Vt = Vt, jobu='A', jobvt='A')
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")

    # 打印结果
    print("\nSingular values (Σ):")
    print(np.array(s).flatten())

    print("\nLeft singular vectors (U):")
    print(np.array(U).reshape(3,3))

    print("\nRight singular vectors transposed (Vᵗ):")
    print(np.array(Vt).reshape(2,2))


def test_lapack_potrf():
    print("==== Running test_lapack_potrf ====")
    
    # 对称正定矩阵 A（3x3）
    A = matrix([
    [3.0, 1.0, 0.0],
    [1.0, 3.0, 1.0],
    [0.0, 1.0, 3.0]
    ], tc='d')

    # print("原始矩阵 A：")
    # print(A)

    # 注意：lapack.potrf 是 **in-place** 的，调用后 A 会被覆盖为其 Cholesky 下三角矩阵
    # lapack.potrf(A)

    t1 = time.time()
    # 执行 SVD: A = U * diag(s) * Vt
    lapack.potrf(A)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")


def test_lapack_ormqr():

    # Step 1: A (3x2) — generate reflectors
    A_np = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], dtype=np.float64)

    # Step 2: C must be (2x2) — Q(3x2) @ C(2x2) => result (3x2)
    C_np = np.array([
        [7.0, 8.0],
        [9.0, 10.0]
    ], dtype=np.float64)

    # CVXOPT matrices (column-major)
    A = matrix(A_np, tc='d')
    tau = matrix(0.0, (2,1), tc='d')
    C = matrix(C_np, tc='d')  # shape (2,2)

    # LAPACK geqrf → get Q reflectors
    lapack.geqrf(A, tau)

    # Create output buffer large enough for Q @ C → (3x2)
    # Since C gets overwritten, we allocate 3x2 explicitly
    C_full = matrix(0.0, (3,2), tc='d')
    # Copy input C into top part (to match expected layout)
    for j in range(2):
        for i in range(2):
            C_full[i,j] = C[i,j]

    # Apply Q @ C
    lapack.ormqr(A, tau, C_full, side='L', trans='N')

    # Convert to numpy
    C_result = np.array(C_full).reshape((3, 2), order='F')

    # Compare with NumPy
    Q_np, _ = qr(A_np, mode='economic')  # (3x2)
    C_expected = Q_np @ C_np  # (3x2)

    print("===== CVXOPT result (Q @ C) =====")
    print(np.round(C_result, 6))
    print("===== NumPy expected result =====")
    print(np.round(C_expected, 6))

    assert np.allclose(C_result, C_expected, atol=1e-6)
    print("✅ Test PASSED")


def test_lapack_geqrf():
    print("==== Running test_lapack_geqrf ====")
    
    # 定义原始矩阵 A（列主序，和 C 一致）
    A_np = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], dtype=np.float64)

    # 转为 CVXOPT matrix（列主序！）
    A = matrix(A_np, tc='d')

    # 准备 tau 向量（长度 = min(m, n) = 2）
    tau = matrix(0.0, (2,1), tc='d')

    # 调用 lapack.geqrf（就地修改 A）
    lapack.geqrf(A, tau)

    # 打印 QR 分解结果
    print("After lapack.geqrf:")
    print("A (R upper + reflectors lower):")
    print(np.array(A).reshape((3,2), order='F'))

    print("tau:")
    print(np.array(tau).flatten())
    
def test_lapack_trtrs():
    print("==== Running test_lapack_trtrs ====")

    # 定义下三角矩阵 A (3x3)，列主序：每列按行填充
    A_data = [
        1.0, 2.0, 4.0,   # col 0
        0.0, 3.0, 5.0,   # col 1
        0.0, 0.0, 6.0    # col 2
    ]
    A = matrix(A_data, (3, 3), 'd')

    # 定义右端项矩阵 B (3x2)
    B_data = [
        1.0, 8.0, 32.0,  # col 0 (rhs 1)
        2.0, 13.0, 47.0  # col 1 (rhs 2)
    ]
    B = matrix(B_data, (3, 2), 'd')
    
    n = 600               # 方阵 A 的大小
    nrhs = 100            # 右端向量个数

    # === 生成随机下三角矩阵 A ===
    A_np = np.tril(np.random.randn(n, n))
    
    # === 生成随机右端矩阵 B ===
    B_np = np.random.randn(n, nrhs)

    # === 转为 CVXOPT 矩阵（列主序）===
    A = matrix(A_np, tc='d')
    B = matrix(B_np, tc='d')

    
    t1 = time.time()
    # 调用 lapack.trtrs 解 A X = B，B 将被原地修改为解 X
    lapack.trtrs(A, B)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")

    t1 = time.time()
    for i in range(N):
        lapack.trtrs(A, B)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.6f} seconds")
    

    # 打印结果 X（存储在 B 中）
    print("Solution X:")
    for i in range(3):
        for j in range(2):
            print(f"{B[i,j]:8.4f}", end=' ')
        print()

    
def test_lapack_all():
    print("==== Running test_lapack ====")
    
    # test_lapack_gesvd()
    # test_lapack_potrf()
    # test_lapack_ormqr()
    # test_lapack_geqrf()
    
    test_lapack_trtrs()  # Uncomment to run this test
    
    # Test base_emul
    # test_base_emul()  # Uncomment if you want to run this test
    
    # Test base_ediv
    # test_base_ediv()  # Uncomment if you want to run this test
    # test_base_sqrt()  # Uncomment if you want to run this test


if __name__ == "__main__":
    test_lapack_all()
