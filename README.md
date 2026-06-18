# Project Documentation
GRIN Ray-Tracing Simulator
=========================

This project simulates ray propagation through graded-index (GRIN) media and lens profiles and compares numerical integrators. 
- compares Euler and RK4 integration for a linear GRIN profile
- evaluates trajectory differences using RMSE and max position error
- simulates ray bundles through Maxwell Fish‑Eye and Luneburg lenses
- checks the Bouguer invariant for rotationally symmetric propagation
- saves ray and lens plots as image files

## Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main demonstration:

```bash
python main.py
```

This executes the default simulation and generates plots in the project root. Plots are shown by default; to disable plotting, edit the `show=True` arguments in `main.py`.

## `physics.py`

Defines ray propagation physics for different refractive-index profiles and computes state derivatives used by the integrator.

### State representation
- `state = [x, y, px, py]`
  - `x`, `y`: ray position
  - `px`, `py`: momentum-like direction components

The core ray equations are:
- `dx/ds = px / n`
- `dy/ds = py / n`
- `dpx/ds = ∂n/∂x`
- `dpy/ds = ∂n/∂y`

### Models included
- linear GRIN profile via `full_state_derivative`
- Maxwell Fish-Eye lens via `derivatives_fisheye`
- Luneburg lens via `derivatives_luneburg`
- `bouguer_invariant(state)` for conservation checks

## `integrators.py`

Implements the numerical integration methods.

### `euler_step(state, deriv_fn, ds)`
- explicit Euler integration
- `S_{n+1} = S_n + ds * f(S_n)`

### `rk4_step(state, deriv_fn, ds)`
- classical 4th-order Runge-Kutta
- more accurate than Euler for curved ray paths

## `simulate.py`

Collects a full ray trajectory over repeated integration steps.

### `run_simulation(initial_state, deriv_fn, integrator, ds, n_steps)`
- returns a trajectory array of shape `(n_steps + 1, 4)`
- advances the ray state step-by-step using the selected integrator

## `analysis.py`

Provides trajectory comparison metrics.

### Summary metrics
- `compute_rmse(reference, target)` for all state components
- `compute_position_rmse(reference, target)` for position only
- `compute_momentum_rmse(reference, target)` for momentum only
- `max_position_error(reference, target)` for the largest position deviation
- `compare_trajectories(reference, target)` returns all of the above

## `plots.py`

Contains plotting helpers for ray and lens visualizations.

### Plot functions
- `plot_comparison_trajectories(...)` overlays multiple ray trajectories
- `plot_momentum_evolution(...)` plots momentum components over steps
- `plot_position_evolution(...)` plots position coordinates over steps
- `plot_lens_trajectories(...)` visualizes ray bundles through a lens region

## `main.py`

This is the top-level driver script.

### What it does
- sets up a sample initial ray for the linear GRIN model
- compares Euler and RK4 trajectories at a coarse step size
- compares RK4 and Euler at a finer step size
- simulates ray bundles through Maxwell Fish-Eye and Luneburg lenses
- prints final states and error summaries
- checks the Bouguer invariant for the RK4 and Euler trajectories
- generates and saves comparison plots and lens ray figures

### Running the project
- `python main.py` executes all default simulations
- output includes printed metrics and saved PNG plot files

## 
- Focused on the ray-tracing core: GRIN propagation, integrator comparison, and lens visualization
- it advances a single ray through the linear GRIN medium using three integrators
- it also traces multiple rays through Maxwell Fish-Eye and Luneburg lens profiles
- it prints numeric summaries and error comparisons to the console
- it generates and saves plot images in the project root
- the script shows plots interactively by default, unless you edit `show=False`


---