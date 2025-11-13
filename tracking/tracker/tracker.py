"""
Multi-Object Tracker with Data Association.

Manages multiple tracks and associates detections to existing tracks using
Mahalanobis distance with gating (simple but effective approach).
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from .track import Track, reset_track_id_counter


class MultiObjectTracker:
    """
    Multi-Object Tracker using Kalman Filters and simple data association.

    The tracker maintains a list of active tracks and performs:
    1. Prediction: Predict all tracks to current frame
    2. Association: Match detections to tracks using Mahalanobis distance
    3. Update: Update matched tracks with detections
    4. Management: Create new tracks, delete old tracks
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        min_age: int = 3,
        max_distance: float = 9.21,
        max_x: float = 5000.0,
        max_y: float = 3000.0,
        q_pos: float = 50.0,
        q_vel: float = 100.0,
        r_pos: float = 100.0,
        sigma_pos_init: float = 500.0,
        sigma_vel_init: float = 1000.0,
    ):
        """
        Initialize the Multi-Object Tracker.

        Args:
            max_age: Maximum frames without update before deleting track (default: 5)
            min_hits: Minimum hits for track confirmation (default: 3)
            min_age: Minimum age for track confirmation (default: 3)
            max_distance: Maximum Mahalanobis distance for association (default: 9.21)
                         9.21 corresponds to 99% confidence for 2D (chi-squared)
            max_x: Maximum valid x position in mm (default: 5000)
            max_y: Maximum valid y position in mm (default: 3000)
            q_pos: Process noise std dev for position [mm] (default: 50)
            q_vel: Process noise std dev for velocity [mm/s] (default: 100)
            r_pos: Measurement noise std dev for position [mm] (default: 100)
            sigma_pos_init: Initial position uncertainty [mm] (default: 500)
            sigma_vel_init: Initial velocity uncertainty [mm/s] (default: 1000)
            
        Note: dt is now passed dynamically to update() based on actual timestamps.
        """
        # Track management parameters
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_age = min_age
        self.max_distance = max_distance
        self.max_x = max_x
        self.max_y = max_y

        # Kalman filter parameters
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos = r_pos
        self.sigma_pos_init = sigma_pos_init
        self.sigma_vel_init = sigma_vel_init

        # List of active tracks
        self.tracks: List[Track] = []

        # Statistics
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

    def update(self, detections: List[Dict], dt: float = 0.1) -> List[Dict]:
        """
        Main tracking update function. Call this once per frame with new detections.

        Performs the complete tracking cycle:
        1. Predict all tracks
        2. Associate detections to tracks
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Delete old tracks
        6. Return confirmed tracks

        Args:
            detections: List of detection dictionaries, each with:
                - 'center': {'x': float, 'y': float} in mm
                - 'class_id': int
                - 'score': float (0-1)
                - 'width': float (optional)
            dt: Time step in seconds since last update (default: 0.1s)
                Should be calculated from actual timestamps for accuracy.

        Returns:
            List of confirmed track dictionaries (from track.to_dict())
        """
        self.frame_count += 1

        # Step 1: Predict all existing tracks
        self._predict_tracks(dt)

        # Step 2: Associate detections to tracks
        (
            matched_tracks,
            matched_detections,
            unmatched_tracks,
            unmatched_detections,
        ) = self._associate(detections)

        # Step 3: Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].update(detections[det_idx])

        # Step 4: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 5: Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])

        # Step 6: Delete old/invalid tracks
        self._delete_old_tracks()

        # Step 7: Return only confirmed tracks
        return self.get_confirmed_tracks()

    def _predict_tracks(self, dt: float) -> None:
        """
        Predict all active tracks to the current frame.
        
        Args:
            dt: Time step in seconds
        """
        for track in self.tracks:
            track.predict(dt)

    def _associate(
        self, detections: List[Dict]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Associate detections to tracks using Mahalanobis distance with gating.

        Simple greedy association algorithm:
        1. Compute distance matrix (all tracks vs all detections)
        2. Apply gating (reject associations with distance > threshold)
        3. Greedy matching: assign closest detection to each track

        Args:
            detections: List of detection dictionaries

        Returns:
            Tuple of:
                - matched_track_indices: List of track indices that were matched
                - matched_detection_indices: List of detection indices that were matched
                - unmatched_track_indices: List of track indices without match
                - unmatched_detection_indices: List of detection indices without match
        """
        if len(self.tracks) == 0:
            # No tracks: all detections are unmatched
            return [], [], [], list(range(len(detections)))

        if len(detections) == 0:
            # No detections: all tracks are unmatched
            return [], [], list(range(len(self.tracks))), []

        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(detections)

        # Apply gating: distances > threshold become invalid
        distance_matrix[distance_matrix > self.max_distance] = np.inf

        # Greedy matching
        matched_tracks = []
        matched_detections = []
        used_detections = set()

        # For each track, find the closest detection
        for track_idx in range(len(self.tracks)):
            # Get distances for this track to all detections
            distances = distance_matrix[track_idx, :]

            # Find minimum distance (excluding already used detections)
            min_dist = np.inf
            best_det_idx = -1

            for det_idx in range(len(detections)):
                if det_idx not in used_detections and distances[det_idx] < min_dist:
                    min_dist = distances[det_idx]
                    best_det_idx = det_idx

            # If a valid match was found
            if best_det_idx >= 0 and min_dist < np.inf:
                matched_tracks.append(track_idx)
                matched_detections.append(best_det_idx)
                used_detections.add(best_det_idx)

        # Find unmatched tracks and detections
        unmatched_tracks = [
            i for i in range(len(self.tracks)) if i not in matched_tracks
        ]
        unmatched_detections = [
            i for i in range(len(detections)) if i not in used_detections
        ]

        return (
            matched_tracks,
            matched_detections,
            unmatched_tracks,
            unmatched_detections,
        )

    def _compute_distance_matrix(self, detections: List[Dict]) -> np.ndarray:
        """
        Compute Mahalanobis distance matrix between tracks and detections.

        Mahalanobis distance accounts for uncertainty in the prediction.
        It's better than Euclidean distance because it considers the covariance.

        Formula: d² = (z - ẑ)ᵀ S⁻¹ (z - ẑ)
        where:
            z = actual measurement
            ẑ = predicted measurement (H @ s_predict)
            S = innovation covariance (H @ Σ_predict @ Hᵀ + R)

        Args:
            detections: List of detection dictionaries

        Returns:
            Distance matrix of shape (num_tracks, num_detections)
            Each entry [i, j] is the Mahalanobis distance from track i to detection j
        """
        num_tracks = len(self.tracks)
        num_detections = len(detections)

        distance_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

        for i, track in enumerate(self.tracks):
            # Get predicted measurement and innovation covariance
            z_pred = track.get_predicted_measurement()  # [x, y]

            # Get innovation covariance S = H @ Σ @ H^T + R
            # For our case: H extracts position from state, so
            # S = Σ_pos + R where Σ_pos is the 2x2 position covariance
            state, cov = track.get_state()
            S = cov[0:2, 0:2] + track.R  # Position covariance + measurement noise

            # Compute distance to each detection
            for j, detection in enumerate(detections):
                # Measurement
                z = np.array(
                    [detection["center"]["x"], detection["center"]["y"]],
                    dtype=np.float32,
                )

                # Innovation (residual)
                y = z - z_pred

                # Mahalanobis distance: d² = yᵀ S⁻¹ y
                # We use d (not d²) for easier interpretation
                try:
                    distance_squared = y.T @ np.linalg.inv(S) @ y
                    distance = np.sqrt(distance_squared)
                except np.linalg.LinAlgError:
                    # If S is singular, fall back to Euclidean distance
                    distance = np.linalg.norm(y)

                distance_matrix[i, j] = distance

        return distance_matrix

    def _create_track(self, detection: Dict) -> None:
        """
        Create a new track from an unmatched detection.

        Args:
            detection: Detection dictionary
        """
        new_track = Track(
            detection=detection,
            q_pos=self.q_pos,
            q_vel=self.q_vel,
            r_pos=self.r_pos,
            sigma_pos_init=self.sigma_pos_init,
            sigma_vel_init=self.sigma_vel_init,
        )
        self.tracks.append(new_track)
        self.total_tracks_created += 1

    def _delete_old_tracks(self) -> None:
        """
        Delete tracks that should no longer be tracked.

        Tracks are deleted if:
        - Not updated for too long (time_since_update > max_age)
        - Outside valid tracking area
        """
        tracks_to_keep = []

        for track in self.tracks:
            if track.should_be_deleted(
                max_age=self.max_age, max_x=self.max_x, max_y=self.max_y
            ):
                self.total_tracks_deleted += 1
            else:
                tracks_to_keep.append(track)

        self.tracks = tracks_to_keep

    def get_confirmed_tracks(self) -> List[Dict]:
        """
        Get all confirmed tracks as dictionaries.

        Only confirmed tracks (hits >= min_hits, age >= min_age) are returned.
        These are the tracks that should be published/used.

        Returns:
            List of track dictionaries (from track.to_dict())
        """
        confirmed = []
        for track in self.tracks:
            if track.is_confirmed(min_hits=self.min_hits, min_age=self.min_age):
                confirmed.append(track.to_dict())
        return confirmed

    def get_all_tracks(self) -> List[Dict]:
        """
        Get all tracks (including unconfirmed) as dictionaries.

        Useful for debugging/visualization.

        Returns:
            List of all track dictionaries
        """
        return [track.to_dict() for track in self.tracks]

    def get_statistics(self) -> Dict:
        """
        Get tracker statistics for monitoring/debugging.

        Returns:
            Dictionary with statistics
        """
        num_confirmed = sum(
            1 for t in self.tracks if t.is_confirmed(self.min_hits, self.min_age)
        )

        return {
            "frame_count": self.frame_count,
            "active_tracks": len(self.tracks),
            "confirmed_tracks": num_confirmed,
            "total_created": self.total_tracks_created,
            "total_deleted": self.total_tracks_deleted,
        }

    def reset(self) -> None:
        """Reset the tracker (delete all tracks and reset statistics)."""
        self.tracks = []
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
        # Reset global track ID counter
        reset_track_id_counter()


if __name__ == "__main__":
    """
    Test the MultiObjectTracker with a simple scenario.
    """
    print("=" * 60)
    print("Multi-Object Tracker Test")
    print("=" * 60)

    # Create tracker
    tracker = MultiObjectTracker(
        max_age=5,
        min_hits=3,
        max_distance=500.0,  # Use larger threshold for test (mm)
    )

    print(f"\nTracker initialized with parameters:")
    print(f"  max_age={tracker.max_age}, min_hits={tracker.min_hits}")

    # Simulate a scenario with 2 objects moving
    print(f"\n" + "=" * 60)
    print("Scenario: 2 objects moving, 1 gets occluded temporarily")
    print("=" * 60)

    frames = [
        # Frame 1: 2 objects detected
        [
            {
                "class_id": 2,
                "score": 0.95,
                "center": {"x": 1000.0, "y": 500.0},
                "width": 300.0,
            },
            {
                "class_id": 10,
                "score": 0.90,
                "center": {"x": 1500.0, "y": -200.0},
                "width": 200.0,
            },
        ],
        # Frame 2: Both objects moved
        [
            {
                "class_id": 2,
                "score": 0.94,
                "center": {"x": 1050.0, "y": 480.0},
                "width": 300.0,
            },
            {
                "class_id": 10,
                "score": 0.92,
                "center": {"x": 1480.0, "y": -220.0},
                "width": 200.0,
            },
        ],
        # Frame 3: Both objects continue moving
        [
            {
                "class_id": 2,
                "score": 0.96,
                "center": {"x": 1100.0, "y": 460.0},
                "width": 300.0,
            },
            {
                "class_id": 10,
                "score": 0.91,
                "center": {"x": 1460.0, "y": -240.0},
                "width": 200.0,
            },
        ],
        # Frame 4: Object 1 (pedestrian) is occluded! Only vehicle visible
        [
            {
                "class_id": 2,
                "score": 0.95,
                "center": {"x": 1150.0, "y": 440.0},
                "width": 300.0,
            },
        ],
        # Frame 5: Both objects visible again
        [
            {
                "class_id": 2,
                "score": 0.93,
                "center": {"x": 1200.0, "y": 420.0},
                "width": 300.0,
            },
            {
                "class_id": 10,
                "score": 0.89,
                "center": {"x": 1420.0, "y": -280.0},
                "width": 200.0,
            },
        ],
        # Frame 6: Continued tracking
        [
            {
                "class_id": 2,
                "score": 0.94,
                "center": {"x": 1250.0, "y": 400.0},
                "width": 300.0,
            },
            {
                "class_id": 10,
                "score": 0.90,
                "center": {"x": 1400.0, "y": -300.0},
                "width": 200.0,
            },
        ],
    ]

    for frame_idx, detections in enumerate(frames, start=1):
        print(f"\n{'='*60}")
        print(f"Frame {frame_idx}")
        print(f"{'='*60}")
        print(f"Detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(
                f"  Det {i}: class={det['class_id']}, pos=({det['center']['x']:.0f}, {det['center']['y']:.0f})"
            )

        # Update tracker (with dt=0.1 for test)
        confirmed_tracks = tracker.update(detections, dt=0.1)

        # Show all tracks (including unconfirmed)
        all_tracks = tracker.get_all_tracks()
        print(f"\nAll active tracks: {len(all_tracks)}")
        for track in all_tracks:
            confirmed = (
                "✓"
                if track["age"] >= tracker.min_age and track["hits"] >= tracker.min_hits
                else "✗"
            )
            print(
                f"  Track {track['track_id']}: class={track['class_id']}, "
                f"pos=({track['position']['x']:.0f}, {track['position']['y']:.0f}), "
                f"vel=({track['velocity']['vx']:.0f}, {track['velocity']['vy']:.0f}), "
                f"hits={track['hits']}, age={track['age']}, "
                f"time_since_update={track['time_since_update']}, "
                f"confirmed={confirmed}"
            )

        # Show confirmed tracks
        print(f"\nConfirmed tracks: {len(confirmed_tracks)}")
        for track in confirmed_tracks:
            print(
                f"  Track {track['track_id']}: class={track['class_id']}, "
                f"pos=({track['position']['x']:.0f}, {track['position']['y']:.0f}), "
                f"confidence={track['confidence']:.2f}"
            )

    # Show final statistics
    print(f"\n{'='*60}")
    print("Final Statistics")
    print(f"{'='*60}")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)