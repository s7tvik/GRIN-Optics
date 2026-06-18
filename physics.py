import numpy as np

# linear n(y) = n0 + alpha*y
def compute_n(state, n0=1.5, alpha=0.3):
    y = state[1]  # y-coordinate of the ray
    return n0 + alpha * y


def grad_n(state, n0=1.5, alpha=0.3):
    return np.array([0.0, alpha]) 
# dn/dx = 0.0 & dn/dy = alpha 

def full_state_derivative(state, n0=1.5, alpha=0.3):
    # dS/ds = [dx/ds, dy/ds, dpx/ds, dpy/ds]
    x, y, px, py = state
    n = compute_n(state, n0, alpha)
    grad = grad_n(state, n0, alpha)
    return np.array([px/n, py/n, grad[0], grad[1]])


# ── Maxwell's Fish-Eye ────────────────────────────────────────
def n_fisheye(state, n0=2.0, a=1.0, center_x=0.0, center_y=0.0):
    x, y = state[0] - center_x, state[1] - center_y
    r2 = x**2 + y**2
    if r2 <= a**2:
        return n0 / (1 + r2 / a**2)
    return 1.0


def grad_n_fisheye(state, n0=2.0, a=1.0, center_x=0.0, center_y=0.0):
    x, y = state[0] - center_x, state[1] - center_y
    r2 = x**2 + y**2
    if r2 <= a**2 and r2 > 0:
        factor = -2 * n0 / (a**2 * (1 + r2 / a**2)**2)
        return np.array([factor * x, factor * y])
    return np.array([0.0, 0.0])


def derivatives_fisheye(state, n0=2.0, a=1.0, center_x=0.0, center_y=0.0):
    x, y, px, py = state
    n = n_fisheye(state, n0, a, center_x, center_y)
    grad = grad_n_fisheye(state, n0, a, center_x, center_y)
    return np.array([px / n, py / n, grad[0], grad[1]])

# ── Luneburg Lens ─────────────────────────────────────────────
def n_luneburg(state, a=1.0, center_x=0.0, center_y=0.0):
    x, y = state[0] - center_x, state[1] - center_y
    r = np.sqrt(x**2 + y**2)
    if r < a:
        return np.sqrt(2 - (r / a)**2)
    return 1.0   # outside: free space


def grad_n_luneburg(state, a=1.0, center_x=0.0, center_y=0.0):
    x, y = state[0] - center_x, state[1] - center_y
    r = np.sqrt(x**2 + y**2)
    if r >= a or r < 1e-10:
        return np.array([0.0, 0.0])
    n = n_luneburg(state, a, center_x, center_y)
    return np.array([-x / (a**2 * n), -y / (a**2 * n)])


def derivatives_luneburg(state, a=1.0, center_x=0.0, center_y=0.0):
    x, y, px, py = state
    n = n_luneburg(state, a, center_x, center_y)
    grad = grad_n_luneburg(state, a, center_x, center_y)
    return np.array([px / n, py / n, grad[0], grad[1]])

# ── Conservation Quantities ───────────────────────────────────
def bouguer_invariant(state):
    # L = x*py - y*px (should be constant for spherical media)
    x, y, px, py = state
    return x * py - y * px