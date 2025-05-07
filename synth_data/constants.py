import numpy as np

DIM_W = 16
IMG_SHAPE = (32, 32)

a = 0.8
b = 0.3
sigma_X = 0.1
sigma_Z = 0.5

sigma_Y = 0.05
sigma_W = 0.1

beta_1 = 0.3
beta_2 = -0.5
beta_3 = np.zeros(16)
beta_3[0] = -1.0
beta_3[1] = 1.2
beta_3[4] = 2.0
beta_3[8] = -1.0
beta_3[10] = 1.0
beta_3[11] = -0.8


A = np.random.randn(DIM_W, 2) * 0.3
A[0, 0] = 0.2
A[0, 1] = 0.8
A[1, 0] = -1.0
A[1, 1] = 0.5
A[4, 1] = 1.5 
A[8, 0] = 0.25
A[8, 1] = 0.5
A[10, 1] = 1.3
A[11, 1] = 1.0