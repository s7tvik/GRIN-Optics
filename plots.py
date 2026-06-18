
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_trajectory(trajectory, save_path='trajectory.png', title='Ray Trajectory', show=True):
    """
    Plot a ray trajectory in 2D (x-y) space.
    """
    x_positions = trajectory[:, 0]
    y_positions = trajectory[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, y_positions, 'b-', linewidth=2, label='Ray path')
    plt.scatter(x_positions[0], y_positions[0], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x_positions[-1], y_positions[-1], color='red', s=100, label='End', zorder=5)
    
    plt.xlabel('x (arc-length)', fontsize=12)
    plt.ylabel('y (transverse)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved as {save_path}")
    
    if show:
        plt.show()


def plot_comparison_trajectories(trajectories, labels, save_path='comparison_trajectory.png', title='Trajectory Comparison', show=True):
    """
    Plot multiple ray trajectories on the same x-y axes.
    """
    plt.figure(figsize=(10, 6))
    for trajectory, label in zip(trajectories, labels):
        x_positions = trajectory[:, 0]
        y_positions = trajectory[:, 1]
        plt.plot(x_positions, y_positions, linewidth=2, label=label)

    plt.xlabel('x (arc-length)', fontsize=12)
    plt.ylabel('y (transverse)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved as {save_path}")
    if show:
        plt.show()


def plot_momentum_evolution(trajectory, save_path='momentum.png', show=True):
    """
    Plot the evolution of optical momentum (px, py) along the ray path.
    
    Parameters:
    -----------
    trajectory : ndarray, shape (n_steps+1, 4)
        State vector array from run_simulation: [x, y, px, py]
    save_path : str, optional
        Path to save the plot image (default: 'momentum.png')
    show : bool, optional
        Whether to display the plot (default: True)
    """
    steps = np.arange(len(trajectory))
    px = trajectory[:, 2]
    py = trajectory[:, 3]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, px, 'b-', linewidth=2, label='px (x-momentum)')
    plt.plot(steps, py, 'r-', linewidth=2, label='py (y-momentum)')
    
    plt.xlabel('Step number', fontsize=12)
    plt.ylabel('Momentum component', fontsize=12)
    plt.title('Optical Momentum Evolution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved as {save_path}")
    
    if show:
        plt.show()


def plot_lens_trajectories(trajectories, save_path='lens_trajectories.png', title='Lens Ray Trajectories', lens_radius=1.0, lens_center=(0.0, 0.0), show=True):
    """
    Plot multiple lens ray trajectories with a circular lens overlay.
    """
    plt.figure(figsize=(10, 6))
    for trajectory in trajectories:
        x_positions = trajectory[:, 0]
        y_positions = trajectory[:, 1]
        plt.plot(x_positions, y_positions, linewidth=2)

    lens = Circle(lens_center, lens_radius, color='gray', alpha=0.2, label='Lens region')
    plt.gca().add_patch(lens)

    plt.xlabel('x (arc-length)', fontsize=12)
    plt.ylabel('y (transverse)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved as {save_path}")
    if show:
        plt.show()


def plot_position_evolution(trajectory, save_path='position.png', show=True):
    """
    Plot the evolution of position (x, y) along the ray path.
    
    Parameters:
    -----------
    trajectory : ndarray, shape (n_steps+1, 4)
        State vector array from run_simulation: [x, y, px, py]
    save_path : str, optional
        Path to save the plot image (default: 'position.png')
    show : bool, optional
        Whether to display the plot (default: True)
    """
    steps = np.arange(len(trajectory))
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, x, 'b-', linewidth=2, label='x position')
    plt.plot(steps, y, 'r-', linewidth=2, label='y position')
    
    plt.xlabel('Step number', fontsize=12)
    plt.ylabel('Position coordinate', fontsize=12)
    plt.title('Position Evolution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved as {save_path}")
    
    if show:
        plt.show()
