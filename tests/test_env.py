import unittest

import numpy as np

from kinematic_uuv.env import KNOT_MPS, KinematicUUVEnv
from kinematic_uuv.teacher import teacher_action


class KinematicEnvTests(unittest.TestCase):
    def test_reset_observation_and_scene(self):
        env = KinematicUUVEnv(max_steps=20)
        observation, _ = env.reset(seed=0)
        try:
            self.assertEqual(observation["sonar"].shape, (6, 12, 48))
            self.assertEqual(observation["target"].shape, (4,))
            self.assertEqual(observation["path"].shape, (5,))
            self.assertEqual(observation["motion"].shape, (7,))
            for value in observation.values():
                self.assertTrue(np.isfinite(value).all())
            self.assertGreaterEqual(len(env.obstacles), env.obstacle_count_min)
            self.assertLessEqual(len(env.obstacles), env.obstacle_count_max)
        finally:
            env.close()

    def test_kinematic_limits_are_enforced(self):
        env = KinematicUUVEnv(max_steps=200, min_turn_radius=30.0,
                              max_lat_accel=4.0)
        observation, _ = env.reset(seed=1)
        env.cruise_speed_override = 50.0 * KNOT_MPS
        observation, _ = env.reset(seed=1)
        max_lat_accel = 0.0
        try:
            for _ in range(100):
                action = teacher_action(observation)
                observation, _, terminated, truncated, info = env.step(action)
                max_lat_accel = max(
                    max_lat_accel,
                    env.state["speed"] * abs(env.state["yaw_rate"]),
                )
                self.assertLessEqual(env.state["speed"], env.speed_max + 1e-9)
                self.assertLessEqual(abs(env.state["pitch"]), env.max_pitch + 1e-9)
                self.assertLessEqual(max_lat_accel, env.max_lat_accel + 0.05)
                if terminated or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

