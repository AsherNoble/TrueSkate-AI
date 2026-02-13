import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.trajectory_spline import load_trajectory, TrajectorySpline

t, x, y = load_trajectory('your_data.csv')
spline = TrajectorySpline(t, x, y, smoothing=500)  # adjust as needed

# Query at any time
x_pos, y_pos = spline.position(0.5)
vx, vy = spline.velocity(0.5)
speed = spline.speed(0.5)