import logging
import sys
import os
import time
import numpy as np
import cvxpy as cp

from typing import Tuple, Any, Optional


class LMISolver:
    def __init__(self):
        self.alpha = 0.9
        self.hd = 1e-10
        self.phi = 0.15
        self.cc = 0.6
        self.b1 = 1 / 0.8  # yaw
        self.b2 = 1 / (1.0 * self.cc)  # height
        self.b3 = 1 / 1.5  # velocity
        self.c1 = 1 / 25
        self.c2 = 1 / 50

        # fixed matrices
        self.D = np.array([[self.b2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, self.b3, 0, 0, 0, 0, 0],
                           [0, 0, 0, self.b1, 0, 0, 0, 0, 0, 0]])
        self.C = np.array([[self.c1, 0, 0, 0, 0, 0],
                           [0, self.c1, 0, 0, 0, 0],
                           [0, 0, self.c1, 0, 0, 0],
                           [0, 0, 0, self.c2, 0, 0],
                           [0, 0, 0, 0, self.c2, 0],
                           [0, 0, 0, 0, 0, self.c2]])

        # Parameters and variables
        self.z_param = cp.Parameter(10, name='tracking_err')
        self.A_param = cp.Parameter((10, 10), name='A')
        self.B_param = cp.Parameter((10, 6), name='B')

        self.Q = cp.Variable((10, 10), PSD=True)
        self.T = cp.Variable((6, 6), PSD=True)
        self.R = cp.Variable((6, 10))

        Z = cp.diag(self.z_param)
        Q, T, R = self.Q, self.T, self.R
        A, B = self.A_param, self.B_param

        self.problem = cp.Problem(cp.Minimize(0), [
            cp.bmat([[self.alpha * Q, Q @ A.T + R.T @ B.T],
                     [A @ Q + B @ R, Q / (1 + self.phi)]]) >> 0,
            cp.bmat([[Q, R.T],
                     [R, T]]) >> 0,
            Q - 10 * Z @ Z >> 0,
            np.eye(3) - self.D @ Q @ self.D.T >> 0,
            np.eye(6) - self.C @ T @ self.C.T >> 0,
            T - self.hd * np.eye(6) >> 0,
        ])

    def patch_lmi(self, tracking_err: np.ndarray) -> Tuple[Optional[Tuple[Any, Any]], bool]:
        roll, pitch, yaw = tracking_err[1:4]

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        Rzyx = Rz @ Ry @ Rx

        T_s = 1 / 20
        aA = np.zeros((10, 10))
        aA[0, 6] = 1
        aA[1:4, 7:10] = Rzyx
        aB = np.zeros((10, 6))
        aB[4:, :] = np.eye(6)

        A = np.eye(10) + T_s * aA
        B = T_s * aB

        self.z_param.value = tracking_err
        self.A_param.value = A
        self.B_param.value = B

        self.problem.solve(solver=cp.CVXOPT, verbose=False)

        if self.problem.status == 'optimal':
            Q = self.Q.value
            R = self.R.value
            P = np.linalg.inv(Q)
            aF = np.round(aB @ R @ P, 0)
            Fb2 = aF[6:10, 0:4]

            F_kp = -np.block([[np.zeros((2, 6))], [np.zeros((4, 2)), Fb2]])
            F_kd = -aF[4:10, 4:10]

            if np.all(np.linalg.eigvals(P) > 0):
                logging.info("LMIs feasible")
            else:
                print("LMIs infeasible")

            return (F_kp, F_kd), True
        else:
            print("Optimization failed.")
            return None, False


if __name__ == '__main__':
    zz = np.array([0.1, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.16])
    lmisolver = LMISolver()
    t1 = time.time()
    (F_kp, F_kd), _ = lmisolver.patch_lmi(zz)
    t2 = time.time()
    print(f"F_kp is: {F_kp}\nF_kd is: {F_kd}\nconsuming time: {t2 - t1:.4f}s")
    
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
