import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import csv


def load_trajectory(csv_path):
    """Load trajectory data from CSV with columns: t, x, y"""
    t, x, y = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row['t']))
            x.append(float(row['x']))
            y.append(float(row['y']))
    return np.array(t), np.array(x), np.array(y)


class TrajectorySpline:
    def __init__(self, t, x, y, smoothing=None):
        """
        Fit smoothing splines to trajectory data.

        Args:
            t: timestamps
            x: x coordinates
            y: y coordinates
            smoothing: smoothing factor (None = auto, 0 = interpolation, higher = smoother)
        """
        self.t_min = t.min()
        self.t_max = t.max()

        self.spline_x = UnivariateSpline(t, x, s=smoothing)
        self.spline_y = UnivariateSpline(t, y, s=smoothing)

        # Pre-compute derivative splines
        self._dx = self.spline_x.derivative()
        self._dy = self.spline_y.derivative()

    def position(self, t):
        """Get (x, y) position at time t"""
        return self.spline_x(t), self.spline_y(t)

    def velocity(self, t):
        """Get (vx, vy) velocity at time t"""
        return self._dx(t), self._dy(t)

    def speed(self, t):
        """Get scalar speed at time t"""
        vx, vy = self.velocity(t)
        return np.sqrt(vx ** 2 + vy ** 2)


def visualise_fit(t, x, y, smoothing_values):
    """Compare different smoothing factors visually."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for s_val in smoothing_values:
        spline = TrajectorySpline(t, x, y, smoothing=s_val)
        t_smooth = np.linspace(t.min(), t.max(), 500)
        x_fit, y_fit = spline.position(t_smooth)
        speed = spline.speed(t_smooth)

        label = f's={s_val}' if s_val is not None else 's=auto'

        # Spatial path
        axes[0, 0].plot(x_fit, y_fit, label=label, alpha=0.8)

        # X vs time
        axes[0, 1].plot(t_smooth, x_fit, label=label, alpha=0.8)

        # Y vs time
        axes[1, 0].plot(t_smooth, y_fit, label=label, alpha=0.8)

        # Speed vs time
        axes[1, 1].plot(t_smooth, speed, label=label, alpha=0.8)

    # Plot original points
    axes[0, 0].scatter(x, y, c='black', s=10, zorder=5, label='data')
    axes[0, 1].scatter(t, x, c='black', s=10, zorder=5)
    axes[1, 0].scatter(t, y, c='black', s=10, zorder=5)

    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Spatial Path')
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')

    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('x')
    axes[0, 1].set_title('X vs Time')

    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Y vs Time')

    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('speed')
    axes[1, 1].set_title('Speed vs Time')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150)
    plt.show()
    print("Saved plot to trajectory_comparison.png")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python trajectory_spline.py <csv_file> [smoothing_values...]")
        print("Example: python trajectory_spline.py path.csv 0 100 1000")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Parse smoothing values from command line, or use defaults
    if len(sys.argv) > 2:
        smoothing_values = [float(s) if s != 'auto' else None for s in sys.argv[2:]]
    else:
        smoothing_values = [None, 0, 100, 1000]  # Default comparison set

    t, x, y = load_trajectory(csv_path)
    print(f"Loaded {len(t)} points, t range: [{t.min():.3f}, {t.max():.3f}]")

    visualise_fit(t, x, y, smoothing_values)