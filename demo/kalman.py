# Functions for Kalman filter

import numpy as np
import math

# 1R1C
def update_1R1C(P, x_hat, y):
    I = np.identity(3)
    C = np.mat([1, 0, 0])
    R = 1
    # Kalman gain
    #K = P * C.T * np.linalg.inv(C * P * C.T + R) 
    K = P * C.T / (C * P * C.T + R) 

    x_hat = x_hat + K * (y - (C * x_hat))
    P = (I -  K * C) * P
    return x_hat, P

def kalman_1R1C(t_r, t_a, power, time):
    P = np.identity(3)
    H = np.identity(3)
    x_hat = np.mat([[t_r[0]], [1.0e-07], [1.0e-05]])

    #t_r_index = t_r.index
    for i in range(1, len(t_r)):
        y = t_r[i]
        x_hat, P = update_1R1C(P, x_hat, y)
        #time = (t_r_index[i] - t_r_index[i-1]).seconds
        A = np.mat([[1, power[i-1]  * time, -time * (t_r[i-1] - t_a[i-1])],
                    [0, 1, 0],
                    [0, 0, 1]])
        x_hat = A * x_hat
        P = A * P * A.T
    return x_hat

def residual_1R1C(t_r, t_a, power, parameter, time):
    pointIntegral = [t_r[0]]
    c_inv = parameter[0]
    ua_c = parameter[1]

    for i in range(1,  len(t_r)):
        #time = (t_r.index[i] - t_r.index[i-1]).seconds
        wall_in = ua_c * (t_a[i] - t_r[i]) + (power[i] * c_inv) 
        pointIntegral = np.append(pointIntegral, pointIntegral[i - 1] + wall_in * time)
    diff = np.array(t_r) - pointIntegral
    sum_sq = sum(diff ** 2)
    sigma = math.sqrt(sum_sq/len(pointIntegral))
    return sigma, diff

# 2R1C
def update_2R1C(P, x_hat, y):
    I = np.identity(4)
    C = np.mat([1, 0, 0, 0])
    R = 1
    # Kalman gain
    #K = P * C.T * np.linalg.inv(C * P * C.T + R) 
    K = P * C.T / (C * P * C.T + R) 

    x_hat = x_hat + K * (y - (C * x_hat))
    P = (I -  K * C) * P
    return x_hat, P

def kalman_2R1C(t_r, t_a, t_h, power, time):
    P = np.identity(4)
    H = np.identity(4)
    x_hat = np.mat([[t_r[0]], [1.0e-08], [1.0e-06], [1.0e-06]])

    #t_r_index = t_r.index
    for i in range(1, len(t_r)):
        y = t_r[i]
        x_hat, P = update_2R1C(P, x_hat, y)
        #time = (t_r_index[i] - t_r_index[i-1]).seconds
        A = np.mat([[1, power[i-1]  * time,  (t_a[i-1] - t_r[i-1]) * time, (t_h[i-1] - t_r[i-1]) * time],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        x_hat = A * x_hat
        P = A * P * A.T
    return x_hat

def residual_2R1C(t_i, t_a, t_h, power, parameter, time):
    pointIntegral = [t_i[0]]
    c_inv = parameter[0]
    u_ai_c = parameter[1]
    u_hi_c = parameter[2]

    for i in range(1,  len(t_i)):
        #time = (t_i.index[i] - t_i.index[i-1]).seconds
        wall_in = u_ai_c * (t_a[i] - t_i[i]) + u_hi_c * (t_h[i] - t_i[i]) + (power[i] * c_inv) 
        pointIntegral = np.append(pointIntegral, pointIntegral[i - 1] + wall_in * time)
    diff = np.array(t_i) - pointIntegral
    sum_sq = sum(diff ** 2)
    sigma = math.sqrt(sum_sq/len(pointIntegral))
    return sigma, diff


# 3R1C
def update_3R1C(P, x_hat, y):
    I = np.identity(5)
    C = np.mat([1, 0, 0, 0, 0])
    R = 1
    # Kalman gain
    #K = P * C.T * np.linalg.inv(C * P * C.T + R) 
    K = P * C.T / (C * P * C.T + R) 

    x_hat = x_hat + K * (y - (C * x_hat))
    P = (I -  K * C) * P
    return x_hat, P

def kalman_3R1C(t_r, t_a, t_s, t_h, power, time):
    P = np.identity(5)
    H = np.identity(5)
    x_hat = np.mat([[t_r[0]], [1.0e-08], [1.0e-06], [1.0e-06], [1.0e-06]])

    #t_r_index = t_r.index
    for i in range(1, len(t_r)):
        y = t_r[i]
        x_hat, P = update_3R1C(P, x_hat, y)
        #time = (t_r_index[i] - t_r_index[i-1]).seconds
        A = np.mat([[1, power[i-1]  * time,  (t_a[i-1] - t_r[i-1]) * time, (t_h[i-1] - t_r[i-1]) * time, (t_s[i-1] - t_r[i-1]) * time],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])
        x_hat = A * x_hat
        P = A * P * A.T
    return x_hat

def residual_3R1C(t_i, t_a, t_h, t_s, power, parameter, time):
    pointIntegral = [t_i[0]]
    c_inv = parameter[0]
    u_ai_c = parameter[1]
    u_hi_c = parameter[2]
    u_si_c = parameter[3]

    for i in range(1,  len(t_i)):
        wall_in = u_ai_c * (t_a[i] - t_i[i]) + u_hi_c * (t_h[i] - t_i[i]) + u_si_c * (t_s[i] - t_i[i]) + (power[i] * c_inv) 
        pointIntegral = np.append(pointIntegral, pointIntegral[i - 1] + wall_in * time)
    diff = np.array(t_i) - pointIntegral
    sum_sq = sum(diff ** 2)
    sigma = math.sqrt(sum_sq/len(pointIntegral))
    return sigma, diff
