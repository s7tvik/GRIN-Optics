import numpy as np
from physics import full_state_derivative
from integrators import rk4_step

def run_simulation(initial_state, deriv_fn, integrator, ds, n_steps):
    """
    Collect trajectory over n_steps using the given integrator.
    """
    trajectory = np.zeros((n_steps + 1, 4))
    trajectory[0] = initial_state
    state = initial_state.copy()
    
    for i in range(n_steps):
        state = integrator(state, deriv_fn, ds)
        trajectory[i + 1] = state
    
    return trajectory