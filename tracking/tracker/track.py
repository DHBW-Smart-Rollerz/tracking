"""
Track Class for Single Object Tracking.

Each Track represents one detected object being tracked over time using a Kalman Filter.
A track maintains its own state, covariance, and metadata (ID, age, hits, etc.).
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .kalman_filter import KalmanFilter

from collections import defaultdict


# DEPRECATED: Global track ID counter (kept for backwards compatibility)
# New code should pass track_id directly to Track.__init__()
# Each MultiObjectTracker now maintains its own instance-level counter
# _global_track_id_counter = 0


def get_next_track_id() -> int:
    """
    DEPRECATED: Get the next unique track ID from global counter.

    This function is kept for backwards compatibility but should not be used
    in new code. MultiObjectTracker now manages IDs internally.

    Returns:
        Unique track ID
    """
    global _global_track_id_counter
    track_id = _global_track_id_counter
    _global_track_id_counter += 1
    return track_id


def reset_track_id_counter() -> None:
    """
    DEPRECATED: Reset the global track ID counter.

    This function is kept for backwards compatibility but should not be used
    in new code. Use MultiObjectTracker.reset() instead.
    """
    global _global_track_id_counter
    _global_track_id_counter = 0


class Track:
    """
    Represents a single tracked object with Kalman Filter state estimation.

    Each track has:
    - Unique track_id
    - Kalman filter state (position + velocity)
    - Covariance matrix (uncertainty)
    - Metadata: age, hits, time_since_update, class_id, score
    """

    def __init__(
        self,
        detection: Dict,
        track_id: int,
        q_pos: float = None,
        q_vel: float = None,
        r_pos: float = None,
        sigma_pos_init: float = None,
        sigma_vel_init: float = None,
    ):
        """
        Initialize a new track from a detection.

        Args:
            detection: Detection dictionary with keys:
                - 'center': {'x': float, 'y': float} in mm
                - 'class_id': int
                - 'score': float (0-1)
                - 'width': float (optional)
            track_id: Unique track ID (provided by tracker)
            q_pos: Process noise std dev for position [mm]
            q_vel: Process noise std dev for velocity [mm/s]
            r_pos: Measurement noise std dev for position [mm]
            sigma_pos_init: Initial position uncertainty [mm]
            sigma_vel_init: Initial velocity uncertainty [mm/s]

        Note: dt is now passed dynamically to predict() for accurate timing.
        """
        # Assign track ID from tracker
        self.track_id = track_id

        # Store object class and detection score
        self.class_id = detection["class_id"]
        self.score = detection["score"]
        self.width = detection.get("width", 0.0)

        # Initialize Kalman Filter (no dt parameter anymore)
        self.kf = KalmanFilter()

        # Create noise matrices
        self.Q = self.kf.create_process_noise_matrix(q_pos=q_pos, q_vel=q_vel)
        self.R = self.kf.create_measurement_noise_matrix(r_pos=r_pos)

        # Initialize state: [x, y, vx, vy]
        # Initial velocity is 0 (we don't know it yet)
        x = detection["center"]["x"]
        y = detection["center"]["y"]
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)

        # Initialize covariance (high uncertainty for velocity)
        self.covariance = self.kf.create_initial_covariance(
            sigma_pos=sigma_pos_init, sigma_vel=sigma_vel_init
        )

        # Track management properties
        self.age = 0  # Total number of frames since creation
        self.hits = 1  # Number of successful measurement updates
        self.time_since_update = 0  # Frames since last measurement update

        # Store original detection for reference
        self.last_detection = detection

        # Class history for majority voting
        self.class_history = defaultdict(int)
        self.class_history[detection["class_id"]] += 1
        self.class_id = detection["class_id"]

    def compensate_ego_motion(self, dx: float, dy: float, dtheta: float) -> None:
        """
        Transform track state from old ego frame to new ego frame.

        This compensates for the vehicle's own movement between frames.
        Must be called BEFORE predict() each frame.

        The transformation accounts for:
        - Translation: the ego vehicle moved by (dx, dy) in the old frame
        - Rotation: the ego vehicle rotated by dtheta (positive = left turn)

        Objects that are actually static will have near-zero velocity after
        compensation, instead of inheriting the ego vehicle's motion.

        Args:
            dx: Ego displacement in x (forward) in mm, in old frame coordinates
            dy: Ego displacement in y (left) in mm, in old frame coordinates
            dtheta: Ego heading change in radians (positive = counter-clockwise / left)
        """
        if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dtheta) < 1e-9:
            return  # No ego motion, skip transformation

        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)

        # Rotation matrix from old ego frame to new ego frame: R(-dtheta)
        # If ego turns left by dtheta, objects appear to rotate right
        R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)

        # Transform position: subtract ego displacement, then rotate
        pos = self.state[:2] - np.array([dx, dy], dtype=np.float32)
        self.state[:2] = R @ pos

        # Transform velocity: rotation only (no translation component)
        self.state[2:4] = R @ self.state[2:4]

        # Transform covariance: R4 @ Σ @ R4ᵀ where R4 = blockdiag(R, R)
        R4 = np.zeros((4, 4), dtype=np.float32)
        R4[0:2, 0:2] = R
        R4[2:4, 2:4] = R
        self.covariance = R4 @ self.covariance @ R4.T

    def predict(self, dt: float) -> None:
        """
        Predict the next state using the Kalman Filter.

        This should be called once per frame before attempting to match detections.
        Updates self.state and self.covariance with predictions.

        Args:
            dt: Time step in seconds (time since last update)
        """
        self.state, self.covariance = self.kf.predict(
            s=self.state, Sigma=self.covariance, Q=self.Q, dt=dt
        )

        # Increment age (track exists for one more frame)
        self.age += 1

        # Increment time since last update (will be reset if matched)
        self.time_since_update += 1

    def update(self, detection: Dict) -> None:
        """
        Update the track with a matched detection using Kalman Filter.

        Args:
            detection: Detection dictionary with keys:
                - 'center': {'x': float, 'y': float} in mm
                - 'class_id': int
                - 'score': float (0-1)
                - 'width': float (optional)
        """
        # Extract measurement [x, y]
        z = np.array(
            [detection["center"]["x"], detection["center"]["y"]], dtype=np.float32
        )

        # Store innovation norm BEFORE update
        innovation = z - self.kf.get_measurement_prediction(self.state)
        self._last_innovation_norm = float(np.linalg.norm(innovation))

        # Update state and covariance using Kalman Filter
        self.state, self.covariance = self.kf.update(
            s_predict=self.state, Sigma_predict=self.covariance, z=z, R=self.R
        )

        # Update track properties
        self.hits += 1
        self.time_since_update = 0

        # REPLACE strict assignment with Voting Logic
        detected_class = detection["class_id"]
        self.class_history[detected_class] += 1

        # The class_id is the one with the highest count in history
        # (This prevents a single flickering frame from changing the object type)
        self.class_id = max(self.class_history, key=self.class_history.get)

        self.score = detection["score"]

        # Smooth width using exponential moving average (EMA)
        # Prevents flickering from frame-to-frame detector noise
        new_width = detection.get("width", self.width)
        alpha = 0.3  # Smoothing factor: 0=keep old, 1=take new raw value
        self.width = alpha * new_width + (1.0 - alpha) * self.width

        self.last_detection = detection

    def mark_missed(self) -> None:
        """
        Mark that no detection was matched to this track in the current frame.

        Should be called for tracks that weren't updated in the current frame.
        Note: predict() already increments time_since_update, so this method
        is mainly for clarity and future extensions.
        """
        # time_since_update is already incremented in predict()
        # This method exists for clarity and potential future logic
        pass

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current state and covariance.

        Returns:
            Tuple of (state, covariance)
                - state: [x, y, vx, vy] (4,)
                - covariance: (4x4)
        """
        return self.state.copy(), self.covariance.copy()

    def get_position(self) -> Tuple[float, float]:
        """
        Get the current position estimate.

        Returns:
            Tuple of (x, y) in mm
        """
        return float(self.state[0]), float(self.state[1])

    def get_velocity(self) -> Tuple[float, float]:
        """
        Get the current velocity estimate.

        Returns:
            Tuple of (vx, vy) in mm/s
        """
        return float(self.state[2]), float(self.state[3])

    def get_predicted_measurement(self) -> np.ndarray:
        """
        Get the predicted measurement (where we expect to see the object).

        Useful for data association.

        Returns:
            Predicted measurement [x, y] (2,)
        """
        return self.kf.get_measurement_prediction(self.state)

    def get_position_uncertainty(self) -> float:
        """
        Get the average position uncertainty (useful for confidence estimation).

        Returns:
            Average standard deviation of position in mm
        """
        sigma_x = np.sqrt(self.covariance[0, 0])
        sigma_y = np.sqrt(self.covariance[1, 1])
        return float((sigma_x + sigma_y) / 2.0)

    def is_confirmed(self, min_hits: int = 3, min_age: int = 3) -> bool:
        """
        Check if the track is confirmed (stable and reliable).

        A track is confirmed if it has been updated enough times and exists
        long enough. Only confirmed tracks should be published/used.

        Args:
            min_hits: Minimum number of successful updates (default: 3)
            min_age: Minimum age in frames (default: 3)

        Returns:
            True if track is confirmed, False otherwise
        """
        return self.hits >= min_hits and self.age >= min_age

    def should_be_deleted(
        self, max_age: int = 5, max_x: float = 5000.0, max_y: float = 3000.0
    ) -> bool:
        """
        Check if the track should be deleted.

        A track should be deleted if:
        1. It hasn't been updated for too long (lost object)
        2. It has moved outside the valid tracking area

        Args:
            max_age: Maximum frames without update (default: 5)
            max_x: Maximum valid x position in mm (default: 5000)
            max_y: Maximum valid y position in mm (default: 3000)

        Returns:
            True if track should be deleted, False otherwise
        """
        # Check if not updated for too long
        if self.time_since_update > max_age:
            return True

        # Check if outside valid area
        x, y = self.get_position()
        if abs(x) > max_x or abs(y) > max_y:
            return True

        return False

    def to_dict(self) -> Dict:
        """
        Convert track to dictionary format for publishing or logging.

        Returns:
            Dictionary with track information:
                - track_id: int
                - class_id: int
                - position: {'x': float, 'y': float} in mm
                - velocity: {'vx': float, 'vy': float} in mm/s
                - confidence: float (based on uncertainty)
                - age: int
                - hits: int
                - time_since_update: int
                - width: float (from last detection)
        """
        x, y = self.get_position()
        vx, vy = self.get_velocity()

        # Compute confidence based on position uncertainty
        # Lower uncertainty = higher confidence
        # Scale: uncertainty 0-500mm -> confidence 1.0-0.0
        # uncertainty = self.get_position_uncertainty()
        # confidence = max(0.0, min(1.0, 1.0 - uncertainty / 500.0))

        # # Also factor in detection score
        # confidence = (confidence + self.score) / 2.0

        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "position": {"x": x, "y": y},
            "velocity": {"vx": vx, "vy": vy},
            "confidence": self.get_tracking_confidence(),
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "width": self.width,
        }

    def get_tracking_confidence(self) -> float:
        """
        Compute localization confidence: "How sure are we the object is HERE?"

        Combines four signals:
        1. Position covariance (Kalman filter uncertainty)
        2. Innovation magnitude (prediction vs. measurement agreement)
        3. Track maturity (hit ratio)
        4. Staleness penalty (frames without update)

        Returns:
            Confidence in [0, 1], where 1 = highly certain position
        """
        # --- 1. Covariance-based confidence ---
        # Average position std dev from Kalman covariance
        sigma_x = np.sqrt(self.covariance[0, 0])
        sigma_y = np.sqrt(self.covariance[1, 1])
        avg_sigma = (sigma_x + sigma_y) / 2.0

        # Map to [0, 1]: sigma=0 -> 1.0, sigma>=sigma_max -> 0.0
        # sigma_max should match your tracking area scale
        sigma_max = 500.0  # mm — tune this to your scenario
        c_covariance = max(0.0, 1.0 - avg_sigma / sigma_max)

        # --- 2. Innovation-based confidence ---
        # Uses last stored innovation magnitude
        # Small residual = prediction matches reality = good
        if hasattr(self, "_last_innovation_norm"):
            innov_max = 300.0  # mm — expected max reasonable residual
            c_innovation = max(0.0, 1.0 - self._last_innovation_norm / innov_max)
        else:
            c_innovation = 0.5  # neutral if no update yet

        # --- 3. Track maturity ---
        # hit_ratio: fraction of frames where we got a measurement
        hit_ratio = self.hits / max(self.age, 1)
        c_maturity = min(1.0, hit_ratio)  # already in [0, 1]

        # --- 4. Staleness penalty ---
        # Exponential decay for each frame without measurement
        decay_rate = 0.3  # how aggressively to penalize missed frames
        c_staleness = np.exp(-decay_rate * self.time_since_update)

        # --- Weighted combination ---
        # Covariance is the primary signal, the rest modulate it
        confidence = (
            0.50 * c_covariance
            + 0.20 * c_innovation
            + 0.15 * c_maturity
            + 0.15 * c_staleness
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def __repr__(self) -> str:
        """String representation for debugging."""
        x, y = self.get_position()
        vx, vy = self.get_velocity()
        return (
            f"Track(id={self.track_id}, class={self.class_id}, "
            f"pos=({x:.1f}, {y:.1f}), vel=({vx:.1f}, {vy:.1f}), "
            f"hits={self.hits}, age={self.age})"
        )
