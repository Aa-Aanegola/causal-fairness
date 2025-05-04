import numpy as np

DIM_W = 16
IMG_SHAPE = (32, 32)

a = 0.75
b = 0.25
sigma_X = 0.2
sigma_Z = 0.5

# Effects on Y
beta_1 = 1.0
beta_2 = -0.5
beta_3 = np.zeros(16)
beta_3[0] = -1.0
beta_3[1] = 1.0
beta_3[4] = 1.5
beta_3[8] = -0.5

sigma_Y = 0.05
sigma_W = 0.5

A = np.random.randn(DIM_W, 2) * 0.2
A[0, 0] = 1.0
A[1, 0] = -1.0
A[4, 1] = 1.0 
A[8, 0] = 0.5
A[8, 1] = 0.5