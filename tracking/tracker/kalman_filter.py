"""
Kalman Filter Implementation for 2D Object Tracking.

Implements a 2D Constant Velocity Model for tracking objects in autonomous vehicles.

State Vector: s = [x, y, vx, vy]ᵀ
- x, y: Position in mm (vehicle-relative coordinates)
- vx, vy: Velocity in mm/s

Measurement Vector: z = [x, y]ᵀ
- Only position is measured, velocity is estimated by the filter
"""

from typing import Tuple

import numpy as np


class KalmanFilter:
    """
    2D Kalman Filter for object tracking with constant velocity model.
    """

    def __init__(self):
        """
        Initialize the Kalman Filter.
        
        Note: dt is now passed dynamically to predict() for accurate timing.
        """
        self.state_dim = 4  # [x, y, vx, vy]
        self.measurement_dim = 2  # [x, y]

        # Create measurement matrix H
        # H maps state to measurement: z = H @ s
        self.H = np.array(
            [
                [1, 0, 0, 0],  # Measure x directly
                [0, 1, 0, 0],  # Measure y directly (vx, vy are not measured)
            ],
            dtype=np.float32,
        )

        # Identity matrix for update step
        self.I = np.eye(self.state_dim, dtype=np.float32)

    def _create_state_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Create the state transition matrix F for constant velocity model.

        Args:
            dt: Time step in seconds

        Returns:
            F: 4x4 state transition matrix
        """
        F = np.array(
            [
                [1, 0, dt, 0],  # x_new = x + vx*dt
                [0, 1, 0, dt],  # y_new = y + vy*dt
                [0, 0, 1, 0],  # vx stays constant
                [0, 0, 0, 1],  # vy stays constant
            ],
            dtype=np.float32,
        )
        return F

    def predict(
        self, s: np.ndarray, Sigma: np.ndarray, Q: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of the Kalman Filter.

        Implements equations:
            s_predict = F @ s_current
            Σ_predict = F @ Σ_current @ F^T + Q

        Args:
            s: Current state vector [x, y, vx, vy] (4,)
            Sigma: Current state covariance matrix (4x4)
            Q: Process noise covariance matrix (4x4)
            dt: Time step in seconds (time since last update)

        Returns:
            Tuple of:
                - s_predict: Predicted state vector (4,)
                - Sigma_predict: Predicted covariance matrix (4x4)
        """
        # Create state transition matrix F with current dt
        F = self._create_state_transition_matrix(dt)
        
        # Predict state (s_p = F·s_t)
        s_predict = F @ s

        # Predict covariance (Σ_p = F·Σ_t·F^T + Q)
        # Note: Adding process noise Q accounts for model uncertainty
        Sigma_predict = F @ Sigma @ F.T + Q

        return s_predict, Sigma_predict

    def update(
        self,
        s_predict: np.ndarray,
        Sigma_predict: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of the Kalman Filter with a new measurement.

        Args:
            s_predict: Predicted state vector [x, y, vx, vy] (4,)
            Sigma_predict: Predicted covariance matrix (4x4)
            z: Measurement vector [x_measured, y_measured] (2,)
            R: Measurement noise covariance matrix (2x2)

        Returns:
            Tuple of:
                - s_update: Updated state vector (4,)
                - Sigma_update: Updated covariance matrix (4x4)
        """
        # Innovation covariance: S = H @ Σ_p @ H^T + R
        # This is the uncertainty in the measurement space
        S = self.H @ Sigma_predict @ self.H.T + R

        # Kalman Gain
        # K determines how much we trust the measurement vs. the prediction
        K = Sigma_predict @ self.H.T @ np.linalg.inv(S)

        # Innovation (residual): Difference between measurement and prediction
        # y = z_measured - z_predicted, where z_predicted = H @ s_predict
        y = z - (self.H @ s_predict)

        # Update state
        # Note: K @ y is equivalent to K @ (ŝ_m - s_p) when transformed via H
        s_update = s_predict + K @ y

        # Update covariance
        # Alternative stable form: (I - K @ H) @ Σ_predict
        Sigma_update = (self.I - K @ self.H) @ Sigma_predict

        return s_update, Sigma_update

    def get_measurement_prediction(self, s: np.ndarray) -> np.ndarray:
        """
        Get the expected measurement from a state.

        Useful for data association: predicts where we expect to see the object.

        Args:
            s: State vector [x, y, vx, vy] (4,)

        Returns:
            z_expected: Expected measurement [x, y] (2,)
        """
        return self.H @ s

    @staticmethod
    def create_process_noise_matrix(
        q_pos: float = 50.0, q_vel: float = 100.0
    ) -> np.ndarray:
        """
        Create process noise covariance matrix Q.

        Q models the uncertainty in our motion model. Higher values mean
        we trust the model less and rely more on measurements.

        Args:
            q_pos: Process noise standard deviation for position [mm] (default: 50)
            q_vel: Process noise standard deviation for velocity [mm/s] (default: 100)

        Returns:
            Q: Process noise covariance matrix (4x4)
        """
        Q = np.array(
            [
                [q_pos**2, 0, 0, 0],
                [0, q_pos**2, 0, 0],
                [0, 0, q_vel**2, 0],
                [0, 0, 0, q_vel**2],
            ],
            dtype=np.float32,
        )
        return Q

    @staticmethod
    def compute_r_at_distance(distance: float, r_base: float, d_ref: float) -> np.ndarray:
        """
        Compute distance-dependent measurement noise covariance matrix R.

        Scales measurement noise linearly with distance, reflecting that camera-based
        detections have higher positional uncertainty at greater ranges (σ ∝ d).

        Args:
            distance: Euclidean distance to the detection in mm
            r_base: Base measurement noise std dev at reference distance [mm]
            d_ref: Reference distance [mm] at which r_base applies

        Returns:
            R: Distance-scaled measurement noise covariance matrix (2x2)
        """
        r_scaled = r_base * (distance / d_ref)
        r_scaled = max(r_base * 0.1, r_scaled)  # Floor at 10% of r_base
        return np.array([[r_scaled**2, 0], [0, r_scaled**2]], dtype=np.float32)

    @staticmethod
    def create_measurement_noise_matrix(r_pos: float = 100.0) -> np.ndarray:
        """
        Create measurement noise covariance matrix R.

        R models the uncertainty in our measurements (detection noise).
        Higher values mean we trust measurements less.

        Args:
            r_pos: Measurement noise standard deviation for position [mm] (default: 100)

        Returns:
            R: Measurement noise covariance matrix (2x2)
        """
        R = np.array([[r_pos**2, 0], [0, r_pos**2]], dtype=np.float32)
        return R

    @staticmethod
    def create_initial_covariance(
        sigma_pos: float = 500.0, sigma_vel: float = 1000.0
    ) -> np.ndarray:
        """
        Create initial state covariance matrix for a new track.

        Initial covariance should reflect our uncertainty when first detecting an object.
        Position is somewhat certain, but velocity is very uncertain (we don't know it yet).

        Args:
            sigma_pos: Initial position standard deviation [mm] (default: 500)
            sigma_vel: Initial velocity standard deviation [mm/s] (default: 1000)

        Returns:
            Sigma_init: Initial covariance matrix (4x4)
        """
        Sigma_init = np.array(
            [
                [sigma_pos**2, 0, 0, 0],
                [0, sigma_pos**2, 0, 0],
                [0, 0, sigma_vel**2, 0],
                [0, 0, 0, sigma_vel**2],
            ],
            dtype=np.float32,
        )
        return Sigma_init


if __name__ == "__main__":
    """
    Simple test to verify the Kalman Filter implementation.
    """
    print("=" * 60)
    print("Kalman Filter Test")
    print("=" * 60)

    # Initialize filter (no dt parameter anymore)
    dt = 0.1  # 10 Hz
    kf = KalmanFilter()

    print(f"\nState transition matrix F (dt={dt}s):")
    print(kf._create_state_transition_matrix(dt))

    print(f"\nMeasurement matrix H:")
    print(kf.H)

    # Create noise matrices
    Q = kf.create_process_noise_matrix(q_pos=50, q_vel=100)
    R = kf.create_measurement_noise_matrix(r_pos=100)

    print(f"\nProcess noise Q:")
    print(Q)

    print(f"\nMeasurement noise R:")
    print(R)

    # Initialize state: object at (1000mm, 500mm) with velocity (200mm/s, -100mm/s)
    s = np.array([1000.0, 500.0, 200.0, -100.0], dtype=np.float32)
    Sigma = kf.create_initial_covariance(sigma_pos=500, sigma_vel=1000)

    print(f"\n" + "=" * 60)
    print("Simulation: Object moving with constant velocity")
    print("=" * 60)

    print(f"\nInitial state s_0:")
    print(f"  Position: ({s[0]:.1f}, {s[1]:.1f}) mm")
    print(f"  Velocity: ({s[2]:.1f}, {s[3]:.1f}) mm/s")

    # Simulate 5 time steps
    for i in range(1, 6):
        # Predict
        s_pred, Sigma_pred = kf.predict(s, Sigma, Q, dt)

        print(f"\nStep {i} - After Prediction:")
        print(f"  Position: ({s_pred[0]:.1f}, {s_pred[1]:.1f}) mm")
        print(f"  Velocity: ({s_pred[2]:.1f}, {s_pred[3]:.1f}) mm/s")

        # Simulate measurement (with some noise)
        true_pos = s_pred[:2] + np.random.randn(2) * 50  # Add 50mm noise
        z = true_pos

        print(f"  Measurement: ({z[0]:.1f}, {z[1]:.1f}) mm")

        # Update
        s, Sigma = kf.update(s_pred, Sigma_pred, z, R)

        print(f"  After Update:")
        print(f"  Position: ({s[0]:.1f}, {s[1]:.1f}) mm")
        print(f"  Velocity: ({s[2]:.1f}, {s[3]:.1f}) mm/s")
        print(
            f"  Position uncertainty: σ_x={np.sqrt(Sigma[0,0]):.1f}, σ_y={np.sqrt(Sigma[1,1]):.1f} mm"
        )

    print("\n" + "=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)