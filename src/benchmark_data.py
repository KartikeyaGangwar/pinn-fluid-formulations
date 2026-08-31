"""
Benchmark Reference Data and Quantitative Metric Utilities for 2D Lid-Driven Cavity Flow.

References:
- Ghia, Ghia & Shin (1982), JCP 48(3):387-411.
- Botella & Peyret (1998), Computers & Fluids 27(4):421-433.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Vertical centerline (x = 0.5): y-coordinates
GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000
])

# Horizontal centerline (y = 0.5): x-coordinates
GHIA_X = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
    0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000
])

# u-velocity along vertical centerline (x = 0.5) for different Reynolds numbers
GHIA_U = {
    100: np.array([
        1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
        -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000
    ]),
    400: np.array([
        1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
        -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000
    ]),
    1000: np.array([
        1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
        -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000
    ]),
    3200: np.array([
        1.00000, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791, 0.07156,
        -0.04272, -0.08660, -0.24427, -0.41933, -0.32407, -0.30174, -0.29012, -0.27485, 0.00000
    ]),
    5000: np.array([
        1.00000, 0.48223, 0.46120, 0.45992, 0.46036, 0.33556, 0.20087, 0.08183,
        -0.03039, -0.07404, -0.22855, -0.42901, -0.33560, -0.32284, -0.31547, -0.30668, 0.00000
    ])
}

# v-velocity along horizontal centerline (y = 0.5) for different Reynolds numbers
GHIA_V = {
    100: np.array([
        0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.24533, -0.22445,
        0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10090, 0.09233, 0.00000
    ]),
    400: np.array([
        0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598,
        0.05186, 0.30174, 0.30203, 0.27428, 0.20561, 0.18029, 0.16699, 0.15301, 0.00000
    ]),
    1000: np.array([
        0.00000, -0.21388, -0.27669, -0.32726, -0.37095, -0.42665, -0.51536, -0.32627,
        0.02526, 0.37095, 0.33075, 0.37095, 0.29012, 0.27485, 0.24687, 0.21388, 0.00000
    ]),
    3200: np.array([
        0.00000, -0.31966, -0.39017, -0.43406, -0.46467, -0.49012, -0.42447, -0.32038,
        0.00945, 0.40435, 0.38883, 0.37012, 0.32284, 0.30668, 0.29012, 0.26188, 0.00000
    ])
}

# Vortex Benchmark references
VORTEX_BENCHMARKS = {
    100: {
        'primary': {'x': 0.6172, 'y': 0.7344, 'psi_min': -0.10330},
        'bot_right': {'x': 0.9453, 'y': 0.0625, 'psi_max': 1.75e-6},
        'bot_left': {'x': 0.0313, 'y': 0.0313, 'psi_max': 1.75e-6}
    },
    400: {
        'primary': {'x': 0.5547, 'y': 0.6055, 'psi_min': -0.11391},
        'bot_right': {'x': 0.8906, 'y': 0.1250, 'psi_max': 6.42e-4},
        'bot_left': {'x': 0.0508, 'y': 0.0469, 'psi_max': 1.42e-5}
    },
    1000: {
        'primary': {'x': 0.5313, 'y': 0.5625, 'psi_min': -0.11793, 'omega_c': 2.0677},
        'bot_right': {'x': 0.8594, 'y': 0.1094, 'psi_max': 1.75e-3, 'omega_c': -0.362},
        'bot_left': {'x': 0.0859, 'y': 0.0781, 'psi_max': 2.31e-4, 'omega_c': -0.340},
        'top_left': {'x': 0.0625, 'y': 0.9063, 'psi_max': 1.50e-6}
    }
}

def compute_relative_l2_error(pred, ref):
    diff_norm = np.linalg.norm(pred - ref)
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-12:
        return float(diff_norm)
    return float(diff_norm / ref_norm)

def compute_linf_error(pred, ref):
    return float(np.max(np.abs(pred - ref)))

def interpolate_centerlines(X, Y, u, v, x_ghia=GHIA_X, y_ghia=GHIA_Y):
    x_1d = X[0, :]
    y_1d = Y[:, 0]
    interp_u = RegularGridInterpolator((y_1d, x_1d), u, bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((y_1d, x_1d), v, bounds_error=False, fill_value=None)
    pts_u = np.column_stack([y_ghia, np.full_like(y_ghia, 0.5)])
    pts_v = np.column_stack([np.full_like(x_ghia, 0.5), x_ghia])
    return interp_u(pts_u), interp_v(pts_v)

def evaluate_centerline_metrics(u_pred_center, v_pred_center, Re=1000):
    ghia_u_ref = GHIA_U[Re]
    ghia_v_ref = GHIA_V[Re]
    l2_u = compute_relative_l2_error(u_pred_center, ghia_u_ref)
    l2_v = compute_relative_l2_error(v_pred_center, ghia_v_ref)
    linf_u = compute_linf_error(u_pred_center, ghia_u_ref)
    linf_v = compute_linf_error(v_pred_center, ghia_v_ref)
    return {
        'l2_u_centerline': l2_u,
        'l2_v_centerline': l2_v,
        'linf_u_centerline': linf_u,
        'linf_v_centerline': linf_v,
        'combined_l2': 0.5 * (l2_u + l2_v)
    }

def find_vortex_centers(X, Y, psi):
    idx_min = np.unravel_index(np.argmin(psi), psi.shape)
    primary_x = float(X[idx_min])
    primary_y = float(Y[idx_min])
    primary_psi = float(psi[idx_min])
    
    mask_br = (X > 0.6) & (Y < 0.4)
    if np.any(mask_br):
        sub_psi = np.where(mask_br, psi, -1e9)
        idx_br = np.unravel_index(np.argmax(sub_psi), psi.shape)
        br_x = float(X[idx_br])
        br_y = float(Y[idx_br])
        br_psi = float(psi[idx_br])
    else:
        br_x, br_y, br_psi = np.nan, np.nan, np.nan
        
    mask_bl = (X < 0.4) & (Y < 0.4)
    if np.any(mask_bl):
        sub_psi = np.where(mask_bl, psi, -1e9)
        idx_bl = np.unravel_index(np.argmax(sub_psi), psi.shape)
        bl_x = float(X[idx_bl])
        bl_y = float(Y[idx_bl])
        bl_psi = float(psi[idx_bl])
    else:
        bl_x, bl_y, bl_psi = np.nan, np.nan, np.nan
        
    return {
        'primary': {'x': primary_x, 'y': primary_y, 'psi': primary_psi},
        'bot_right': {'x': br_x, 'y': br_y, 'psi': br_psi},
        'bot_left': {'x': bl_x, 'y': bl_y, 'psi': bl_psi}
    }

def compute_integrated_quantities(X, Y, u, v, omega):
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    kinetic_energy = float(0.5 * np.sum((u**2 + v**2)) * dx * dy)
    enstrophy = float(0.5 * np.sum(omega**2) * dx * dy)
    return {
        'kinetic_energy': kinetic_energy,
        'enstrophy': enstrophy
    }
