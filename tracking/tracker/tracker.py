"""
Multi-Object Tracker with Data Association.

Manages multiple tracks and associates detections to existing tracks using
Mahalanobis distance with gating (simple but effective approach).
"""

import time
from typing import Dict, List, Set, Tuple

import numpy as np

from .track import Track

from scipy.optimize import linear_sum_assignment


class MultiObjectTracker:
    """
    Multi-Object Tracker using Kalman Filters and simple data association.

    The tracker maintains a list of active tracks and performs:
    1. Prediction: Predict all tracks to current frame
    2. Association: Match detections to tracks using Mahalanobis distance
    3. Update: Update matched tracks with detec tions
    4. Management: Create new tracks, delete old tracks
    """

    def __init__(
        # Parameters for initialization only, please change in tracking_params.yaml
        self,
        max_age: int = None,
        min_hits: int = None,
        min_age: int = None,
        max_distance: float = None,
        max_x: float = None,
        max_y: float = None,
        q_pos: float = None,
        q_vel: float = None,
        r_pos: float = None,
        sigma_pos_init: float = None,
        sigma_vel_init: float = None,
        id_offset: int = None,
    ):
        """
        Initialize the Multi-Object Tracker.

        Args:
            max_age: Maximum frames without update before deleting track (default: 5)
            min_hits: Minimum hits for track confirmation (default: 3)
            min_age: Minimum age for track confirmation (default: 3)
            max_distance: Maximum Mahalanobis distance for association (default: 3.03)
                         3.03 corresponds to 99% confidence for 2D (chi-squared)
            max_x: Maximum valid x position in mm (default: 5000)
            max_y: Maximum valid y position in mm (default: 3000)
            q_pos: Process noise std dev for position [mm] (default: 50)
            q_vel: Process noise std dev for velocity [mm/s] (default: 100)
            r_pos: Measurement noise std dev for position [mm] (default: 100)
            sigma_pos_init: Initial position uncertainty [mm] (default: 500)
            sigma_vel_init: Initial velocity uncertainty [mm/s] (default: 1000)
            id_offset: Offset for track IDs to avoid conflicts between trackers (default: 0)

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

        # Instance-level track ID counter (prevents ID conflicts between trackers)
        self._next_track_id = id_offset

        # List of active tracks
        # self.tracks: List[Track] = []
        self.tracks: Dict[int, Track] = {}

        # Statistics
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

        self._initial_id_offset = id_offset  # <--- Store this
        self._next_track_id = id_offset

        # Performance timing statistics (running averages)
        self._timing_history = {
            "predict": [],
            "associate": [],
            "update": [],
            "create_delete": [],
            "total": [],
        }
        self._timing_window = 100  # Keep last N measurements for averaging

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
        t0 = time.perf_counter()

        # 1. Predict (Iterieren über values)
        for track in self.tracks.values():
            track.predict(dt)
        t1 = time.perf_counter()

        # 2. Associate
        # Wir übergeben das Dict, aber die Logik innen muss angepasst werden
        matched_ids, matched_dets, unmatched_ids, unmatched_dets = self._associate(
            detections
        )
        t2 = time.perf_counter()

        # 3. Update matched tracks (Zugriff über ID ist jetzt O(1) und sicher!)
        for track_id, det_idx in zip(matched_ids, matched_dets):
            self.tracks[track_id].update(detections[det_idx])

        # 4. Mark missed (Zugriff über ID)
        for track_id in unmatched_ids:
            self.tracks[track_id].mark_missed()
        t3 = time.perf_counter()

        # 5. Create new tracks
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # 6. Delete old tracks (Viel sicherer mit Dict!)
        # Wir sammeln erst die IDs, die gelöscht werden müssen
        ids_to_delete = []
        for track_id, track in self.tracks.items():
            if track.should_be_deleted(self.max_age, self.max_x, self.max_y):
                ids_to_delete.append(track_id)

        # Dann löschen (ohne die Iteration kaputt zu machen)
        for track_id in ids_to_delete:
            del self.tracks[track_id]
            self.total_tracks_deleted += 1
        t4 = time.perf_counter()

        # Record timing (in microseconds for precision)
        timings = {
            "predict": (t1 - t0) * 1e6,
            "associate": (t2 - t1) * 1e6,
            "update": (t3 - t2) * 1e6,
            "create_delete": (t4 - t3) * 1e6,
            "total": (t4 - t0) * 1e6,
        }
        for key, value in timings.items():
            history = self._timing_history[key]
            history.append(value)
            if len(history) > self._timing_window:
                history.pop(0)

        return self.get_confirmed_tracks()

    def _predict_tracks(self, dt: float) -> None:
        """
        Predict all active tracks to the current frame.
        """
        # WICHTIG: .values() hinzufügen!
        for track in self.tracks.values():
            track.predict(dt)

    def _associate(
        self, detections: List[Dict]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian Algorithm (Munkres) with gating.
        """
        track_ids = list(self.tracks.keys())

        if len(track_ids) == 0:
            return [], [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], [], track_ids, []
        # 1. Distanz-Matrix berechnen
        # Matrix berechnen (Achtung: _compute_distance_matrix muss jetzt track_ids nehmen!)
        distance_matrix = self._compute_distance_matrix(detections, track_ids)

        # 2. Matrix für Scipy vorbereiten (Scipy mag kein np.inf)
        # Wir ersetzen unendliche Kosten durch einen sehr hohen Wert,
        # der garantiert über dem Gating-Threshold liegt.
        # z.B. max_distance * 2 oder einfach 1e6
        large_value = 1e6
        cost_matrix = np.nan_to_num(distance_matrix, posinf=large_value)

        # 3. Ungarischer Algorithmus (Globale Optimierung)
        # row_indices sind Track-Indizes, col_indices sind Detection-Indizes
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # matched_tracks = []
        # matched_detections = []

        matched_track_ids = []  # Achtung: IDs, keine Indizes mehr!
        matched_det_indices = []

        # Sets für schnelles Lookup der unmatched
        unmatched_track_ids_set = set(track_ids)
        unmatched_det_indices_set = set(range(len(detections)))

        for r, c in zip(row_indices, col_indices):
            if distance_matrix[r, c] <= self.max_distance:
                # HIER IST DER TRICK:
                # Wir wandeln Matrix-Zeile 'r' zurück in echte 'track_id'
                actual_track_id = track_ids[r]

                matched_track_ids.append(actual_track_id)
                matched_det_indices.append(c)

                if actual_track_id in unmatched_track_ids_set:
                    unmatched_track_ids_set.remove(actual_track_id)
                if c in unmatched_det_indices_set:
                    unmatched_det_indices_set.remove(c)

        return (
            matched_track_ids,
            matched_det_indices,
            list(unmatched_track_ids_set),
            list(unmatched_det_indices_set),
        )

    def _compute_distance_matrix(
        self, detections: List[Dict], track_ids: List[int]
    ) -> np.ndarray:
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

        # HARD LIMIT: e.g., 1.0 meters (1000mm)
        # No object jumps 1 meter in 0.1s unless your velocity model is very wrong
        MAX_EUCLIDEAN_DISTANCE = 1000.0

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]  # Zugriff per Dict Key
            # Get predicted measurement and innovation covariance
            z_pred = track.get_predicted_measurement()  # [x, y]

            # Get innovation covariance S = H @ Σ @ H^T + R
            # For our case: H extracts position from state, so
            # S = Σ_pos + R where Σ_pos is the 2x2 position covariance
            state, cov = track.get_state()
            S = cov[0:2, 0:2] + track.R  # Position covariance + measurement noise

            # Compute distance to each detection
            for j, detection in enumerate(detections):
                # ADD THIS: Hard Gating on Class ID
                # If the detection class is different from track class, set distance to Infinity
                # (Unless you want to allow class switching, but we just established that causes bugs)

                if track.class_id != detection["class_id"]:
                    distance_matrix[i, j] = np.inf
                    continue
                # Measurement
                z = np.array(
                    [detection["center"]["x"], detection["center"]["y"]],
                    dtype=np.float32,
                )

                # Innovation (residual)
                y = z - z_pred

                # 1. Calculate simple Euclidean distance
                euclidean_dist = np.linalg.norm(y)

                # 2. THE FIX: Immediate rejection based on physical distance
                if euclidean_dist > MAX_EUCLIDEAN_DISTANCE:
                    distance_matrix[i, j] = np.inf
                    continue

                # 3. If it passes physical check, do the smart Mahalanobis math
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
        # Get next unique ID from this tracker's counter
        track_id = self._next_track_id
        self._next_track_id += 1

        new_track = Track(
            detection=detection,
            track_id=track_id,
            q_pos=self.q_pos,
            q_vel=self.q_vel,
            r_pos=self.r_pos,
            sigma_pos_init=self.sigma_pos_init,
            sigma_vel_init=self.sigma_vel_init,
        )
        # self.tracks.append(new_track)
        # self.total_tracks_created += 1
        # Neu: Speichern im Dictionary unter der ID
        self.tracks[track_id] = new_track
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
        # WICHTIG: .values() hinzufügen!
        for track in self.tracks.values():
            if track.is_confirmed(min_hits=self.min_hits, min_age=self.min_age):
                confirmed.append(track.to_dict())
        return confirmed

    def get_all_tracks(self) -> List[Dict]:
        """
        Get all tracks (including unconfirmed) as dictionaries.
        """
        # WICHTIG: .values() hinzufügen!
        return [track.to_dict() for track in self.tracks.values()]

    def get_statistics(self) -> Dict:
        """
        Get tracker statistics for monitoring/debugging.

        Returns:
            Dictionary with statistics including performance timing.
        """
        # WICHTIG: .values() hinzufügen!
        num_confirmed = sum(
            1
            for t in self.tracks.values()
            if t.is_confirmed(self.min_hits, self.min_age)
        )

        stats = {
            "frame_count": self.frame_count,
            "active_tracks": len(self.tracks),
            "confirmed_tracks": num_confirmed,
            "total_created": self.total_tracks_created,
            "total_deleted": self.total_tracks_deleted,
        }

        # Add timing statistics (averages over recent frames)
        if self._timing_history["total"]:
            import numpy as np

            for key, history in self._timing_history.items():
                if history:
                    arr = np.array(history)
                    stats[f"timing_{key}_avg_us"] = float(np.mean(arr))
                    stats[f"timing_{key}_max_us"] = float(np.max(arr))
                    stats[f"timing_{key}_last_us"] = history[-1]

        return stats

    def reset(self, reset_id_counter: bool = True) -> None:
        """
        Reset the tracker (delete all tracks and reset statistics).

        Args:
            reset_id_counter: If True, reset the ID counter to id_offset (default: True)
                             Set to False if you want to preserve continuous ID numbering
        """
        self.tracks = []
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

        # Reset instance-level ID counter (not global!)
        if reset_id_counter:
            # Reset to initial offset (preserves separation between trackers)
            self._next_track_id = self._initial_id_offset


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
        id_offset=0,  # Start IDs from 0
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
