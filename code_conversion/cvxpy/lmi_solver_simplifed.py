import logging
import sys
import os
import time
sys.path.insert(0, os.path.abspath("python"))
sys.path.insert(0, os.path.abspath("python/cvxopt"))

import asyncio
import numpy as np
# import python.cvxpy as cp
import cvxpy as cp

from typing import Tuple, Any, Optional

from omegaconf import DictConfig


class LMISolver:
    def __init__(self):
        pass

    @staticmethod
    def patch_lmi(tracking_err, device="cuda:0"):
        """
        A small-scale debug version of patch_lmi.
        tracking_err: [roll, pitch, wx, wy]
        System dimension reduced to 4, input dimension to 2.
        """

        roll = tracking_err[0]
        pitch = tracking_err[1]

        Rzyx = np.eye(2)  # 省略姿态相关旋转矩阵（调试阶段不用）

        alpha = 0.9
        phi = 0.1
        hd = 1e-8

        D = np.matrix([[1.0, 0, 0, 0]])
        C = np.matrix([[1.0, 0],
                    [0, 1.0]])

        T_s = 0.05

        # Continuous-time system (4 states, 2 inputs)
        aA = np.zeros((4, 4))
        aA[0, 2] = 1
        aA[1, 3] = 1

        aB = np.zeros((4, 2))
        aB[2:, :] = np.eye(2)

        A = np.eye(4) + T_s * aA
        B = T_s * aB

        Z = np.diag(tracking_err[:4])

        Q = cp.Variable((4, 4), PSD=True)
        T = cp.Variable((2, 2), PSD=True)
        R = cp.Variable((2, 4))

        constraints = [
            cp.bmat([[alpha * Q, Q @ A.T + R.T @ B.T],
                    [A @ Q + B @ R, Q / (1 + phi)]]) >> 0,
            np.identity(1) - D @ Q @ D.T >> 0,
            np.identity(2) - C @ T @ C.T >> 0
        ]

        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve(solver=cp.CVXOPT, verbose=False)

        if problem.status == 'optimal':
            optimal_Q = Q.value
            optimal_R = R.value
            P = np.linalg.inv(optimal_Q)
            aF = aB @ optimal_R @ P

            F_kp = -np.zeros((2, 4))
            F_kd = -aF[2:4, 2:4]

            return (F_kp, F_kd), True, problem._solve_time
        else:
            print("Optimization failed.")
            return None, False, problem._solve_time




if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])
    roll, pitch, yaw = 0.0, 0.0, -0.000

    zz = np.array([0.1, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.16]) * 1.0
    # zz = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])   # Raw data

    # zz = np.array([0.0073, -0.4821, -0.1257, 0.0309, 0.1311, 0.2719, 0.0342, 0.4687, -0.3929, -0.6300])

    # zz = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])  # Raw data

    # zz = np.array([0.01, 0.00, -0.01, -0.00, 0.4, 0.03, 0.03, -0.08, -0.08, 0.08])

    lmisolver = LMISolver()
    t1 = time.time()
    (F_kp, F_kd), _, _ = lmisolver.patch_lmi(zz)
    t2 = time.time()
    print(f"consuming time1: {t2 - t1}")
    
    REPEAT = 1  # Test counts
    solve_time = 0.0
    total_time = 0.0

    for _ in range(REPEAT):
        t1 = time.time()
        (F_kp, F_kd), _, elapsed_time = lmisolver.patch_lmi(zz)
        t2 = time.time()
        solve_time += elapsed_time
        total_time += (t2 - t1)

    avg_time = total_time / REPEAT * 1e3
    avg_solve_time = solve_time / REPEAT * 1e3
    print(f"Average solution time over {REPEAT} runs: {avg_time:.6f} milliseconds")
    print(f"Average solve time over {REPEAT} runs: {avg_solve_time:.6f} milliseconds")
    
    # zz = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])
    # t1 = time.time()
    # (F_kp, F_kd), _ = lmisolver.patch_lmi(zz)
    # t2 = time.time()
    # print(f"consuming time2: {t2 - t1}")
    
    # zz = np.array([0.01, 0.00, -0.01, -0.00, 0.4, 0.03, 0.03, -0.08, -0.08, 0.08])
    # t1 = time.time()
    # (F_kp, F_kd), _ = lmisolver.patch_lmi(zz)
    # t2 = time.time()
    # print(f"consuming time2: {t2 - t1}")
    
    # zz = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624]) 
    # t1 = time.time()
    # (F_kp, F_kd), _ = lmisolver.patch_lmi(zz)
    # t2 = time.time()
    # print(f"consuming time2: {t2 - t1}")
    
    # ha_teacher = HATeacher()
    # K = ha_teacher.feedback_law(0, 0, 0)
    # print(K)

    # testN = 100
    # s = time.time()
    # for i in range(testN):
    #     F_kp, F_kd = HATeacher.system_patch(0., 0., 0.)
    # e = time.time()
    # duration = (e - s) / testN
    print(f"F_kp is: {F_kp}")
    print(f"F_kd is: {F_kd}")
    # print(f"time: {duration}")
