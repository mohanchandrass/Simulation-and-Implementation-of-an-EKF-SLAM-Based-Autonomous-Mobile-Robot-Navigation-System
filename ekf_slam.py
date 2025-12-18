"""
ekf_slam.py - Enhanced EKF-SLAM Implementation
Efficient landmark re-observation updates using EKF-consistent confidence weighting
"""

import numpy as np
from math import sin, cos, sqrt, atan2


def wrap_to_pi(angle):
    """Wraps an angle in radians to the interval [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class EKFSLAM:
    def __init__(self,
                 init_pose,
                 init_cov=np.diag([0.1**2, 0.1**2, np.deg2rad(1.0)**2]),
                 motion_noise=np.diag([0.05**2, 0.05**2, np.deg2rad(1.0)**2]),
                 sigma_r=0.15,
                 sigma_phi=np.deg2rad(2.0),
                 max_association_dist=50.0):

        # State: [x, y, theta, l1x, l1y, ..., lNx, lNy]
        self.state = np.array(init_pose, dtype=float)
        self.P = np.array(init_cov, dtype=float)

        self.Q_motion = np.array(motion_noise, dtype=float)
        self.R_measure = np.diag([sigma_r**2, sigma_phi**2])

        # Landmark bookkeeping
        self.id_to_index = {}
        self.landmark_observation_count = {}

        self.max_association_dist = max_association_dist
        self.min_observations_for_trust = 3

        # Stats
        self.total_predictions = 0
        self.total_updates = 0
        self.rejected_observations = 0

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    def predict(self, control, dt):
        vc, wc = control
        rx, ry, rth = self.state[0:3]

        # Differential-drive motion model
        x_new = rx + vc * dt * cos(rth)
        y_new = ry + vc * dt * sin(rth)
        th_new = wrap_to_pi(rth + wc * dt)

        self.state[0:3] = [x_new, y_new, th_new]

        # Jacobian wrt robot state
        G_r = np.array([
            [1, 0, -vc * dt * sin(rth)],
            [0, 1,  vc * dt * cos(rth)],
            [0, 0, 1]
        ])

        n = len(self.state)
        G = np.eye(n)
        G[0:3, 0:3] = G_r

        self.P = G @ self.P @ G.T
        self.P[0:3, 0:3] += self.Q_motion
        self.P = 0.5 * (self.P + self.P.T)

        self.total_predictions += 1

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    def update(self, observations):
        if not observations:
            return

        for r_meas, phi_meas, lm_id in observations:
            z = np.array([r_meas, phi_meas])

            if not self._is_valid_observation(z):
                self.rejected_observations += 1
                continue

            if lm_id not in self.id_to_index:
                self._init_landmark(z, lm_id)
                self.landmark_observation_count[lm_id] = 1
            else:
                obs_count = self.landmark_observation_count[lm_id]
                self._update_associated(z, lm_id, obs_count)
                self.landmark_observation_count[lm_id] = obs_count + 1

        self.total_updates += 1

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def _is_valid_observation(self, z):
        r, phi = z
        return (0.0 < r < 500.0) and (-np.pi <= phi <= np.pi)

    # ------------------------------------------------------------------
    # LANDMARK INITIALIZATION
    # ------------------------------------------------------------------
    def _init_landmark(self, z, lm_id):
        r, phi = z
        rx, ry, rth = self.state[0:3]

        bearing = wrap_to_pi(rth + phi)
        lx = rx + r * cos(bearing)
        ly = ry + r * sin(bearing)

        self.state = np.hstack([self.state, lx, ly])

        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[:n_old, :n_old] = self.P

        J_xr = np.array([
            [1, 0, -r * sin(bearing)],
            [0, 1,  r * cos(bearing)]
        ])

        J_z = np.array([
            [cos(bearing), -r * sin(bearing)],
            [sin(bearing),  r * cos(bearing)]
        ])

        P_rr = self.P[0:3, 0:3]
        P_mm = J_xr @ P_rr @ J_xr.T + J_z @ self.R_measure @ J_z.T
        P_xm = self.P[:, 0:3] @ J_xr.T

        P_new[n_old:, n_old:] = P_mm
        P_new[:n_old, n_old:] = P_xm
        P_new[n_old:, :n_old] = P_xm.T

        self.P = P_new
        self.id_to_index[lm_id] = (len(self.state) - 5) // 2

    # ------------------------------------------------------------------
    # LANDMARK UPDATE (RE-OBSERVATION FIX)
    # ------------------------------------------------------------------
    def _update_associated(self, z, lm_id, obs_count):
        r_meas, phi_meas = z
        rx, ry, rth = self.state[0:3]

        lm_idx = self.id_to_index[lm_id]
        lm_state_idx = 3 + 2 * lm_idx
        lx, ly = self.state[lm_state_idx:lm_state_idx + 2]

        dx = lx - rx
        dy = ly - ry
        q = dx**2 + dy**2
        if q < 1e-6:
            return

        r_hat = sqrt(q)
        phi_hat = wrap_to_pi(atan2(dy, dx) - rth)
        z_hat = np.array([r_hat, phi_hat])

        innov = z - z_hat
        innov[1] = wrap_to_pi(innov[1])

        H_r = np.array([
            [-dx / r_hat, -dy / r_hat, 0],
            [ dy / q,     -dx / q,    -1]
        ])

        H_l = np.array([
            [ dx / r_hat, dy / r_hat],
            [-dy / q,     dx / q]
        ])

        n = len(self.state)
        H = np.zeros((2, n))
        H[:, 0:3] = H_r
        H[:, lm_state_idx:lm_state_idx + 2] = H_l

        # Adaptive EKF measurement scaling (key fix)
        confidence_scale = min(1.0, 3.0 / obs_count)
        R_eff = self.R_measure / confidence_scale

        S = H @ self.P @ H.T + R_eff
        mahal = innov.T @ np.linalg.inv(S) @ innov

        if mahal > 9.21:
            self.rejected_observations += 1
            return

        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ innov
        self.state[2] = wrap_to_pi(self.state[2])

        I = np.eye(n)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # GETTERS
    # ------------------------------------------------------------------
    def get_state(self):
        return self.state.copy()

    def get_covariance(self):
        return self.P.copy()

    def get_robot_pose(self):
        return self.state[0:3].copy()

    def get_landmarks(self):
        landmarks = []
        for i in range((len(self.state) - 3) // 2):
            landmarks.append(self.state[3 + 2*i:3 + 2*i + 2])
        return landmarks

    def get_trusted_landmarks(self):
        trusted = []
        for lm_id, idx in self.id_to_index.items():
            if self.landmark_observation_count[lm_id] >= self.min_observations_for_trust:
                pos = self.state[3 + 2*idx:3 + 2*idx + 2]
                trusted.append((lm_id, pos))
        return trusted

    def get_statistics(self):
        return {
            "predictions": self.total_predictions,
            "updates": self.total_updates,
            "rejected_obs": self.rejected_observations,
            "landmarks_mapped": len(self.id_to_index),
            "trusted_landmarks": len(self.get_trusted_landmarks())
        }
