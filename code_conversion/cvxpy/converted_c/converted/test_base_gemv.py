import ctypes
import numpy as np
import time

# import pdb
# pdb.set_trace()

# 加载 C 库
lib = ctypes.CDLL('./build/libbase.so')

# 定义 matrix struct（需与你的 C struct 对齐）
class Matrix(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("nrows", ctypes.c_int),
        ("ncols", ctypes.c_int),
        ("id", ctypes.c_int),
        ("shape", ctypes.c_int * 2),
        ("strides", ctypes.c_int * 2),
        ("ob_exports", ctypes.c_int),
    ]

# 常量
DOUBLE = 1

# 初始化数据
m, n = 500, 500
A_np = np.random.rand(m, n).astype(np.float64)
x_np = np.random.rand(n).astype(np.float64)
y_np = np.zeros(m, dtype=np.float64)

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



A = make_matrix(A_np, m, n)
x = make_matrix(x_np.reshape((n, 1)), n, 1)
y = make_matrix(y_np.reshape((m, 1)), m, 1)

# from cvxopt import matrix
# A = matrix(A_np)
# x = matrix(x_np)
# y = matrix(y_np)

# 设置参数
alpha = ctypes.c_double(1.0)
beta = ctypes.c_double(0.0)

# 绑定函数签名
# lib.base_gemv.argtypes = [ctypes.c_void_p, ctypes.POINTER(Matrix), ctypes.POINTER(Matrix),
#                           ctypes.c_char, ctypes.c_void_p, ctypes.c_void_p,
#                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
#                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.base_gemv.argtypes = [
    ctypes.POINTER(Matrix),  # A
    ctypes.POINTER(Matrix),  # x
    ctypes.POINTER(Matrix),  # y
    ctypes.c_char,           # trans
    ctypes.POINTER(ctypes.c_double),  # alpha
    ctypes.POINTER(ctypes.c_double),  # beta
    ctypes.c_int, ctypes.c_int,       # m, n
    ctypes.c_int, ctypes.c_int,       # incx, incy
    ctypes.c_int, ctypes.c_int, ctypes.c_int  # offsetx, offsety, offsetA
]


lib.base_gemv.restype = ctypes.c_int


# 运行 C 函数
start = time.time()
lib.base_gemv(ctypes.byref(A), ctypes.byref(x), ctypes.byref(y),
              b'N', ctypes.byref(alpha), ctypes.byref(beta),
              -1, -1, 1, 1, 0, 0, 0)
end = time.time()
print("C gemv time:", end - start)

# 用 numpy 比较
start_np = time.time()
y_np2 = alpha.value * A_np @ x_np + beta.value * y_np
end_np = time.time()
print(f"answer from numpy is: {y_np2}")
print("Numpy gemv time:", end_np - start_np)


# 比较结果一致性
print("Max diff:", np.max(np.abs(y_np - y_np2)))
