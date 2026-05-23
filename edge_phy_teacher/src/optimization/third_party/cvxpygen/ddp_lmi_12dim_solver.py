#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import sys
import time
import importlib
import logging
import cvxpy as cp
from cvxpygen import cpg
import numpy as np

np.set_printoptions(suppress=True)

def cvxpy_ddp(random=False, threshold=0.1):
    '''
    Write your DDP-compliant Python Code here
    '''

    solver = 'CVXOPT'
    
    # For testing purposes, you can set random=True to generate random tracking errors within the specified threshold
    if random:
        tracking_err = np.random.uniform(-threshold, threshold, size=12)
    else:
        tracking_err = np.array([0.0, 0.0, 0.01, 0.1, 0.1, 0., 0., 0., 0., 0., 0.1, -0.1])

    roll = tracking_err[3]
    pitch = tracking_err[4]
    yaw = tracking_err[5]

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    Rzyx = Rz.dot(Ry.dot(Rx))
  

    zz_param = cp.Parameter(12, name='tracking_err_square')
    A_param = cp.Parameter((12, 12), name='A')
    B_param = cp.Parameter((12, 6), name='B')

    # Sampling period
    Ts = 1 / 40  # work in 25 to 30

    alpha = 0.9
    hd = 1e-10
    
        
    # State
    b11 = 1 / 10000.0
    b12 = 1 / 10000.0
    b1 = 1 / 0.25         # height
    b2 = 1 / 10000.0      # roll
    b3 = 1 / 10000.0      # pitch
    b4 = 1 / 10000.0      # yaw
    b5 = 1 / 5.0          # vx
    b6 = 1 / 5.0          # vy
    b7 = 1 / 10000.0      # vz
    b8 = 1 / 15.0         # roll rate
    b9 = 1 / 15.0         # pitch rate
    b10 = 1 / 5.0        # yaw rate
    
    # constraint matrices
    D = np.diag([b11, b12, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10])
  

    # System matrices (continuous-time)
    aA = np.zeros((12, 12))
    aA[0:3, 6:9] = np.eye(3)
    aA[3:6, 9:12] = Rzyx
    aB = np.zeros((12, 6))
    aB[6:, :] = np.eye(6)
    
    # aA = aA.T
    # # ac = aA.reshape(-1)
    # ac = aA.flatten()   
    # print('cccc',ac.shape)
    # print('dddd', ac[111])
    # print('dddd', ac[112])
    # print('dddd', ac[113])
    # print('dddd', ac[123])
    # print('dddd', ac[124])
    # print('dddd', ac[125])
    # print('dddd', ac[135])
    # print('dddd', ac[136])
    # print('dddd', ac[137])
    
    # System matrices (discrete-time)
    B = aB * Ts
    A = np.eye(12) + Ts * aA
    Z = cp.diag(zz_param) 

    Q = cp.Variable((12, 12), PSD=True, name='Q')
    R = cp.Variable((6, 12), name='R')

    # Constraints
    constraints = [
        cp.bmat([[alpha * Q, (A_param @ Q).T + (B_param @ R).T],
                 [A_param @ Q + B_param @ R, Q]]) >> 0,
        Q - 12 * Z >> 0,
        np.identity(12) - D @ Q @ D.T >> 0,
    ]
    
    # Define problem and objective
    problem = cp.Problem(cp.Minimize(0), constraints)
    
    # Check if the problem is DPP compliant
    if not problem.is_dpp():
        raise ValueError("Problem is not DPP compliant. Please reformulate the problem" 
                         "to ensure it is DPP compliant.")

    # Set parameter values
    A_param.value = A
    B_param.value = B
    zz_param.value = tracking_err ** 2
    
    # Solve the problem
    problem.solve(solver=solver, verbose=False)
    
    if problem.status == 'optimal':
        optimal_Q = Q.value
        optimal_R = R.value

        # print(f"Optimal Q:\n{optimal_Q}\n")
        # print(f"Optimal R:\n{optimal_R}\n")

        P = np.linalg.inv(optimal_Q)
        
        aF = np.round(aB @ optimal_R @ P, 4)
        
        print(f"F:\n{aF.shape}\n")
        
        F_kp = -aF[6:12, 0:6]
        F_kd = -aF[6:12, 6:12]
        res = (F_kp, F_kd)
        
        
        is_solved = True

    # Failed to solve LMIs
    else:
        res = None
        is_solved = False
        
    return problem, res, is_solved, problem._solve_time
    

def benchmark_ddp(solver_func, repeat=100, warmup=5, random=False, threshold=0.1):
    """
    Benchmark solver performance.

    Parameters:
        repeat  : number of measured runs
        warmup  : warm-up runs (not counted)
    """

    print("Running warm-up...")
    for _ in range(warmup):
        solver_func(random=random, threshold=threshold)

    print("Benchmarking...")

    solve_times = []
    total_times = []

    for _ in range(repeat):
        t1 = time.perf_counter()
        _, _, _, solve_time = solver_func(random=random, threshold=threshold)
        t2 = time.perf_counter()

        total_times.append(t2 - t1)
        solve_times.append(solve_time)

    solve_times = np.array(solve_times) * 1e3  # ms
    total_times = np.array(total_times) * 1e3  # ms

    print("\n==== Benchmark Results ====")
    print(f"Runs: {repeat}")
    print(f"Solve Time (ms):")
    print(f"  Mean: {solve_times.mean():.4f}")
    print(f"  Std : {solve_times.std():.4f}")
    print(f"  Min : {solve_times.min():.4f}")
    print(f"  Max : {solve_times.max():.4f}")

    print(f"\nTotal Time (ms):")
    print(f"  Mean: {total_times.mean():.4f}")
    print(f"  Std : {total_times.std():.4f}")
    print(f"  Min : {total_times.min():.4f}")
    print(f"  Max : {total_times.max():.4f}")

    print("============================\n")


if __name__ == "__main__":

    # Test the DDP solver
    t1 = time.perf_counter()
    problem, res, is_solved, elapsed_time = cvxpy_ddp()
    t2 = time.perf_counter()
    print(f"Elapsed time: {t2 - t1:.4f} seconds")
    if is_solved:
        print("Optimization successful.")
        print(f"F_kp: {res[0]}")
        print(f"F_kd: {res[1]}")
    else:
        print("Optimization failed.")
        
    # Benchmark
    # benchmark_ddp(cvxpy_ddp, repeat=100, warmup=5, random=False)

    