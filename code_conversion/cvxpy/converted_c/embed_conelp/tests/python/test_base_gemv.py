import ctypes
import numpy as np
import time

# 加载 C 库
lib = ctypes.CDLL('./build/libbase.so')

# 定义 matrix struct（需与你的 C struct 对齐）
class Matrix(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("nrows", ctypes.c_int),
        ("ncols", ctypes.c_int),
        ("id", ctypes.c_int),
        ("shape", ctypes.c_int * 2),
        ("strides", ctypes.c_int * 2),
        ("ob_exports", ctypes.c_int)
    ]

DOUBLE = 1
m, n = 900, 250
num_runs = 10000  # 运行次数

# 构造列主序数据
A_np = np.array(np.random.rand(m, n), dtype=np.float64, order='F')
x_np = np.array(np.random.rand(n), dtype=np.float64, order='F')
y_np = np.array(np.zeros(m), dtype=np.float64, order='F')  # output buffer for C

# 构造 Matrix 结构体
def make_matrix(arr, nrows, ncols):
    return Matrix(
        buffer=arr.ctypes.data_as(ctypes.c_void_p),
        nrows=nrows,
        ncols=ncols,
        id=DOUBLE,
        shape=(nrows, ncols),
        strides=(1, nrows),
        ob_exports=0
    )

alpha = ctypes.c_double(1.0)
beta = ctypes.c_double(0.0)

lib.base_gemv.argtypes = [
    ctypes.POINTER(Matrix), ctypes.POINTER(Matrix), ctypes.POINTER(Matrix),
    ctypes.c_char,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.base_gemv.restype = ctypes.c_int

# 平均时间测量：C
c_total = 0.0
for _ in range(num_runs):
    A = make_matrix(A_np, m, n)
    x = make_matrix(x_np.reshape((n, 1)), n, 1)
    y = make_matrix(y_np.reshape((m, 1)), m, 1)
    start = time.time()
    lib.base_gemv(ctypes.byref(A), ctypes.byref(x), ctypes.byref(y),
                  b'N', ctypes.byref(alpha), ctypes.byref(beta),
                  -1, -1, 1, 1, 0, 0, 0)
    c_total += time.time() - start

# 平均时间测量：NumPy
np_total = 0.0
for _ in range(num_runs):
    start = time.time()
    y_np2 = alpha.value * A_np @ x_np + beta.value * y_np
    np_total += time.time() - start

# 输出结果
print(f"Average C gemv time over {num_runs} runs:    {c_total / num_runs:.6f} sec")
print(f"Average NumPy gemv time over {num_runs} runs: {np_total / num_runs:.6f} sec")

# 差异比较
print("Max diff:", np.max(np.abs(y_np - y_np2)))
print("First 5 elements from C gemv:", y_np[:5])
print("First 5 elements from NumPy :", y_np2[:5])
