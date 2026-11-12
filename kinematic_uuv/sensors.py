"""Geometric three-view sonar proxy for the kinematic environment."""
import numpy as np


class SonarArray:
    def __init__(self, elevation_bins=12, azimuth_bins=48,
                 range_min=1.0, range_max=150.0):
        self.elevation_bins = int(elevation_bins)
        self.azimuth_bins = int(azimuth_bins)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self._views = {
            "front": (0.0, 120.0, 30.0),
            "left": (-90.0, 120.0, 8.0),
            "right": (90.0, 120.0, 8.0),
        }
        self._angles = {
            name: (
                np.deg2rad(np.linspace(-elevation / 2.0, elevation / 2.0,
                                       self.elevation_bins)),
                np.deg2rad(np.linspace(yaw - azimuth / 2.0,
                                       yaw + azimuth / 2.0,
                                       self.azimuth_bins)),
            )
            for name, (yaw, azimuth, elevation) in self._views.items()
        }

    @property
    def observation_shape(self):
        return (6, self.elevation_bins, self.azimuth_bins)

    @staticmethod
    def _rotation_matrix(yaw, pitch, roll=0.0):
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        return np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ])

    def _view(self, eta, obstacles, view_name):
        elevation, azimuth = self._angles[view_name]
        elevation_grid, azimuth_grid = np.meshgrid(elevation, azimuth, indexing="ij")
        cos_elevation = np.cos(elevation_grid)
        local_directions = np.stack((
            cos_elevation * np.cos(azimuth_grid),
            cos_elevation * np.sin(azimuth_grid),
            np.sin(elevation_grid),
        ), axis=-1)
        rotation = self._rotation_matrix(eta[5], eta[4])
        directions = (local_directions @ rotation.T).reshape(-1, 3)
        nearest = np.full(len(directions), self.range_max, dtype=np.float32)
        origin = np.asarray(eta[:3], dtype=float)
        for center, radius in obstacles:
            offset = origin - np.asarray(center, dtype=float)
            b = directions @ offset
            c = float(np.dot(offset, offset) - radius * radius)
            discriminant = b * b - c
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            near = -b - root
            far = -b + root
            distance = np.where(near >= 0.0, near, far)
            hits = valid & (distance >= self.range_min)
            nearest[hits] = np.minimum(nearest[hits], distance[hits])
        distances = nearest.reshape(self.elevation_bins, self.azimuth_bins)
        hit = (distances < self.range_max).astype(np.float32)
        normalized = np.clip(
            (distances - self.range_min) / (self.range_max - self.range_min),
            0.0, 1.0,
        )
        return normalized, hit

    def observe(self, eta, obstacles):
        channels = []
        for name in ("front", "left", "right"):
            distance, hit = self._view(eta, obstacles, name)
            channels.extend((distance, hit))
        return np.asarray(channels, dtype=np.float32)

    def near_obstacle(self, sonar, front_limit=45.0, side_limit=25.0):
        front_distance = sonar[0]
        front_hit = sonar[1] > 0.5
        central = slice(max(0, self.azimuth_bins // 2 - self.azimuth_bins // 6),
                        min(self.azimuth_bins,
                            self.azimuth_bins // 2 + self.azimuth_bins // 6))
        normalized_front = (front_limit - self.range_min) / (
            self.range_max - self.range_min
        )
        if np.any(front_hit[:, central] & (front_distance[:, central] <= normalized_front)):
            return True
        side_threshold = (side_limit - self.range_min) / (
            self.range_max - self.range_min
        )
        return bool(np.any((sonar[[3, 5]] > 0.5)
                           & (sonar[[2, 4]] <= side_threshold)))
