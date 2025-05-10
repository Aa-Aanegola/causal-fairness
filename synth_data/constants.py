import numpy as np

DIM_W = 16
IMG_SHAPE = (32, 32)
EFFECT_SCALE = 2

a = 0.2  # reduce Z's dependence on U_xz
b = 0.8
sigma_X = 0.1
sigma_Z = 0.6

sigma_Y = 0.05
sigma_W = 0.3

beta_2 = 0.0  # remove direct Z effect to reduce spurious influence
beta_D = 1.0

beta_3_low = np.random.randn(DIM_W) * 0.1
beta_3_high = np.random.randn(DIM_W) * 0.3

# Shared and group-specific trait effects
beta_3_low[0] = -0.4
beta_3_low[4] = 0.8
beta_3_low[8] = -0.6
beta_3_low[11] = -0.8

beta_3_high[1] = 0.8
beta_3_high[4] = 1.3
beta_3_high[10] = 1.0
beta_3_high[11] = -0.8

# Gamma controls: increase contrast for direct effect
# Slightly increase gamma_1 for stronger direct effect
gamma_0 = -0.21 * EFFECT_SCALE
gamma_1 = 0.35 * EFFECT_SCALE

# Unified A matrix: now 3 x DIM_W (X, Z, (1 - X))
A = np.random.randn(3, DIM_W) * 0.1
A[0, 1] = 1.2   # X-only influence
A[0, 4] = -1.5
A[0, 10] = 0.6

A[1, 0] = -0.3
A[1, 8] = 0.2
A[1, 11] = -0.2

A[2, 0] = 1.0
A[2, 4] = 0.3
A[2, 8] = -0.3

beta_3_low *= EFFECT_SCALE
beta_3_high *= EFFECT_SCALE
# A remains unscaled to avoid logit saturation
d_threshold = 0.7


d_w = {
    'brightness': 1.0,
    'noise': -1.5,
    'vertical_contrast': -0.8,
    'central_intensity': -1.2,
    'shape_signal': 0.6,
    'circle_strength': 1.0,
    'square_strength': 1.2,
    'diagonal_pattern': 0.9,
    'radial_symmetry': 0.4,
    'x': 0.2,
    'z': -0.2, 
    'x0': 0.2
}