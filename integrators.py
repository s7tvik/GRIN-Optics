import numpy as np

def euler_step(state, deriv_fn, ds):
    # #Sn+1=Sn+ds*f(Sn)  , f(S) = dS/ds
    
    return state + deriv_fn(state) * ds

def rk4_step(state, deriv_fn, ds):
    """
    One RK4 step.
    Returns: new_state
    """
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + (ds/2) * k1)
    k3 = deriv_fn(state + (ds/2) * k2)
    k4 = deriv_fn(state + ds * k3)
    return state + (ds/6) * (k1 + 2*k2 + 2*k3 + k4)


