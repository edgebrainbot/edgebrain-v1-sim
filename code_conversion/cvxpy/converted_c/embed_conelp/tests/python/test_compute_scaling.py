import ctypes
import numpy as np
import time
from cvxopt import misc, matrix

s = np.load("../c/input_matrix/s.npy")
z = np.load("../c/input_matrix/z.npy")
lmbda = np.load("../c/input_matrix/lmbda.npy")

dims = {
    'l': 0,
    'q': [],
    's': [10, 6, 20, 16, 10, 3, 6, 6],
}

s = matrix(s)
z = matrix(z)
lmbda = matrix(lmbda)

W = misc.compute_scaling(s, z, lmbda, dims, None)
t1 = time.time()
for i in range(1):
    W = misc.compute_scaling(s, z, lmbda, dims, None)
t2 = time.time()
print(f"Time taken: {t2 - t1:.6f} seconds")
print(f"Computed scaling d: {W['d']}")
