import numpy as np
import functools
from physics import full_state_derivative, derivatives_fisheye, derivatives_luneburg, bouguer_invariant
from integrators import euler_step, rk4_step
from simulate import run_simulation
from plots import (
    plot_momentum_evolution,
    plot_position_evolution,
    plot_comparison_trajectories,
    plot_lens_trajectories,
)
from analysis import compare_trajectories


def simulate_rays(initial_states, deriv_fn, integrator, ds, n_steps):
    return [run_simulation(state, deriv_fn, integrator, ds, n_steps) for state in initial_states]


def make_fisheye_rays(a=1.0, num_rays=21, x0=-3.0, beam_height=0.9):
    y_positions = np.linspace(-beam_height, beam_height, num_rays)
    return [np.array([x0, y, 1.0, 0.0], dtype=float) for y in y_positions]


def make_luneburg_rays(x0=-3.0, y_span=1.0, num_rays=21):
    y_positions = np.linspace(-y_span, y_span, num_rays)
    return [np.array([x0, y, 1.0, 0.0], dtype=float) for y in y_positions]


# Setup
S0 = np.array([0.0, 0.5, 1.65, 0.0])
deriv_fn = functools.partial(full_state_derivative, n0=1.5, alpha=0.3)

# Run simulations
trajectory_rk4 = run_simulation(S0, deriv_fn, rk4_step, ds=0.01, n_steps=500)
trajectory_euler = run_simulation(S0, deriv_fn, euler_step, ds=0.01, n_steps=500)
trajectory_rk4_fine = run_simulation(S0, deriv_fn, rk4_step, ds=0.005, n_steps=1000)
trajectory_euler_fine = run_simulation(S0, deriv_fn, euler_step, ds=0.005, n_steps=1000)

# Lens demos
lens_center = (0.0, 0.0)
fish_deriv = functools.partial(derivatives_fisheye, n0=2.0, a=1.0, center_x=lens_center[0], center_y=lens_center[1])
lune_deriv = functools.partial(derivatives_luneburg, a=1.0, center_x=lens_center[0], center_y=lens_center[1])
fish_rays = make_fisheye_rays(a=1.0, num_rays=17, x0=-3.0, beam_height=0.8)
lune_rays = make_luneburg_rays(x0=-3.0, y_span=0.8, num_rays=17)
trajectory_fisheye_rays = simulate_rays(fish_rays, fish_deriv, rk4_step, ds=0.0005, n_steps=12000)
trajectory_luneburg_rays = simulate_rays(lune_rays, lune_deriv, rk4_step, ds=0.0005, n_steps=12000)

# Print results
print("Starting state:", trajectory_rk4[0])
print("RK4 final state: ", trajectory_rk4[-1])
print("Euler final state:", trajectory_euler[-1])
print("RK4 fine final state:", trajectory_rk4_fine[-1])
print("Euler fine final state:", trajectory_euler_fine[-1])
print("X traveled (RK4):", trajectory_rk4[-1, 0] - trajectory_rk4[0, 0])
print("X traveled (Euler):", trajectory_euler[-1, 0] - trajectory_euler[0, 0])
print("X traveled (RK4 fine):", trajectory_rk4_fine[-1, 0] - trajectory_rk4_fine[0, 0])
print("X traveled (Euler fine):", trajectory_euler_fine[-1, 0] - trajectory_euler_fine[0, 0])
print("Y final position (RK4):", trajectory_rk4[-1, 1])
print("Y final position (Euler):", trajectory_euler[-1, 1])
print("Y final position (RK4 fine):", trajectory_rk4_fine[-1, 1])
print("Y final position (Euler fine):", trajectory_euler_fine[-1, 1])
print("\nFirst 5 RK4 states:")
for i in range(5):
    print(f"RK4 Step {i}: {trajectory_rk4[i]}")

# Analysis
print("\n--- Analysis summary ---")
comparison = compare_trajectories(trajectory_rk4, trajectory_euler)
comparison_fine = compare_trajectories(trajectory_rk4_fine, trajectory_euler_fine)
print("RMSE full state (coarse):", comparison["rmse_full"])
print("Position RMSE (coarse):", comparison["position_rmse"])
print("Momentum RMSE (coarse):", comparison["momentum_rmse"])
print("Max position error (coarse):", comparison["max_position_error"])
print("\nRMSE full state (fine):", comparison_fine["rmse_full"])
print("Position RMSE (fine):", comparison_fine["position_rmse"])
print("Momentum RMSE (fine):", comparison_fine["momentum_rmse"])
print("Max position error (fine):", comparison_fine["max_position_error"])

# Plot results
print("\n--- Generating plots ---")
plot_comparison_trajectories(
    [trajectory_rk4, trajectory_euler],
    ["RK4", "Euler"],
    save_path="trajectory_comparison.png",
    title="Trajectory Comparison: RK4 vs Euler",
    show=True,
)
plot_comparison_trajectories(
    [trajectory_rk4_fine, trajectory_euler_fine],
    ["RK4 Fine", "Euler Fine"],
    save_path="trajectory_fine_comparison.png",
    title="Trajectory Comparison: RK4 Fine vs Euler Fine",
    show=True,
)
plot_momentum_evolution(trajectory_rk4, save_path="momentum_rk4.png", show=True)
plot_position_evolution(trajectory_rk4, save_path="position_rk4.png", show=True)

# Bouguer invariant diagnostics
bouguer_rk4 = np.array([bouguer_invariant(state) for state in trajectory_rk4])
bouguer_euler = np.array([bouguer_invariant(state) for state in trajectory_euler])
print("\n--- Bouguer invariant check ---")
print("RK4 invariant range:", bouguer_rk4.min(), "to", bouguer_rk4.max())
print("Euler invariant range:", bouguer_euler.min(), "to", bouguer_euler.max())
print("Invariant change is small if the range is narrow.")

# Lens plots
plot_lens_trajectories(
    trajectory_fisheye_rays,
    save_path="fisheye_lens_rays.png",
    title="Maxwell Fish-Eye Lens Ray Paths",
    lens_radius=1.0,
    lens_center=lens_center,
    show=True,
)
plot_lens_trajectories(
    trajectory_luneburg_rays,
    save_path="luneburg_lens_rays.png",
    title="Luneburg Lens Ray Paths",
    lens_radius=1.0,
    lens_center=lens_center,
    show=True,
)
