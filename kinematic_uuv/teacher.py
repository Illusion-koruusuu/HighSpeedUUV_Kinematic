"""Path-following teacher used only for behavior-cloning initialization."""
import numpy as np


def teacher_action(observation):
    path_yaw = float(observation["path"][1])
    path_pitch = float(observation["path"][0])
    cross_track = float(observation["path"][4])
    motion = observation["motion"]
    speed, cruise, speed_min, speed_max = map(float, motion[:4])
    turn_factor = max(0.35, float(np.cos(path_yaw)))
    sonar = observation["sonar"]
    front_hits = sonar[1] > 0.5
    front_distance = sonar[0]
    near_front = bool(np.any(front_hits[:, 16:32]
                             & (front_distance[:, 16:32] < 50.0 / 150.0)))
    side_hits = (sonar[[3, 5]] > 0.5) & (sonar[[2, 4]] < 25.0 / 150.0)
    if near_front or np.any(side_hits):
        turn_factor *= 0.55
    desired_speed = float(np.clip(cruise * turn_factor, speed_min, speed_max))
    speed_action = 2.0 * (desired_speed - speed_min) / max(
        speed_max - speed_min, 1e-6
    ) - 1.0
    return np.array([
        np.clip(speed_action, -1.0, 1.0),
        np.clip(2.0 * path_yaw - 1.2 * cross_track, -1.0, 1.0),
        np.clip(3.0 * path_pitch, -1.0, 1.0),
    ], dtype=np.float32)

