import numpy as np

DIM_W = 16
IMG_SHAPE = (32, 32)

a = 0.3
b = 0.8
sigma_X = 0.1
sigma_Z = 0.6

sigma_Y = 0.05
sigma_W = 0.1

beta_1 = -0.1
beta_2 = 0.4
beta_3 = np.random.randn(DIM_W) * 0.1
beta_3[0] = -0.4
beta_3[1] = 3.0
beta_3[4] = 2.3
beta_3[8] = -1.0
beta_3[10] = 2.6
beta_3[11] = -1.0


A = np.random.randn(DIM_W, 2) * 0.2
A[0, 0] = 1.0
A[0, 1] = 1.3
A[1, 0] = -1.0
A[1, 1] = 2.5
A[4, 1] = 2.0
A[8, 0] = 0.4
A[8, 1] = 0.25
A[10, 1] = 1.3
A[11, 0] = 0.8