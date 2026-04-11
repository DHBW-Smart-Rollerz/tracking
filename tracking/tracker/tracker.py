"""
Multi-Object Tracker with Data Association.

Manages multiple tracks and associates detections to existing tracks using
Mahalanobis distance with gating (simple but effective approach).
"""

import time
from collections import defaultdict
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
        r_dist_ref: float = None,
        sigma_pos_init: float = None,
        sigma_vel_init: float = None,
        id_offset: int = None,
        duplicate_distance: float = 0.0,
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
        self.r_dist_ref = r_dist_ref
        self.sigma_pos_init = sigma_pos_init
        self.sigma_vel_init = sigma_vel_init
        self.duplicate_distance = duplicate_distance

        # List of active tracks
        self.tracks: Dict[int, Track] = {}

        # Statistics
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

        self._initial_id_offset = id_offset  # <--- Store this
        # Instance-level track ID counter (prevents ID conflicts between trackers)
        self._next_track_id = id_offset

        # Performance timing statistics (running averages for real-time display)
        self._timing_history = {
            "predict": [],
            "associate": [],
            "update": [],
            "create_delete": [],
            "deduplicate": [],
            "total": [],
        }
        self._timing_window = 100  # Keep last N measurements for averaging

        # Full timing log for post-run analysis (every frame, never truncated)
        self._timing_log: List[Dict] = []

    def process_step(self, detections: List[Dict], dt: float = 0.1) -> List[Dict]:
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
        matched_ids, matched_dets, _, unmatched_dets = self._associate(detections)
        t2 = time.perf_counter()

        # 3. Update matched tracks (Zugriff über ID ist jetzt O(1) und sicher!)
        for track_id, det_idx in zip(matched_ids, matched_dets):
            self.tracks[track_id].measurement_update(detections[det_idx])

        t3 = time.perf_counter()

        # 4. Create new tracks
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # 5. Delete old tracks (Viel sicherer mit Dict!)
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

        # 6. Deduplicate: remove duplicate tracks of the same class that are too close
        if self.duplicate_distance > 0:
            self._deduplicate_tracks()
        t5 = time.perf_counter()

        # Record timing (in microseconds for precision)
        timings = {
            "predict": (t1 - t0) * 1e6,
            "associate": (t2 - t1) * 1e6,
            "update": (t3 - t2) * 1e6,
            "create_delete": (t4 - t3) * 1e6,
            "deduplicate": (t5 - t4) * 1e6,
            "total": (t5 - t0) * 1e6,
        }
        for key, value in timings.items():
            history = self._timing_history[key]
            history.append(value)
            if len(history) > self._timing_window:
                history.pop(0)

        # Full log entry for post-run analysis
        self._timing_log.append(
            {
                "frame": self.frame_count,
                "timestamp": t0,  # Wall-clock (time.perf_counter seconds)
                "predict_us": timings["predict"],
                "associate_us": timings["associate"],
                "update_us": timings["update"],
                "create_delete_us": timings["create_delete"],
                "deduplicate_us": timings["deduplicate"],
                "total_us": timings["total"],
                "num_tracks": len(self.tracks),
                "num_detections": len(detections),
            }
        )

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
        Associate detections to tracks using Hungarian Algorithm per class_id.

        Tracks and detections are grouped by class_id first, then the Hungarian
        Algorithm runs separately for each class. This eliminates cross-class
        matches by design rather than by penalizing them.
        """
        track_ids = list(self.tracks.keys())

        if len(track_ids) == 0:
            return [], [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], [], track_ids, []

        # Group tracks and detections by class_id
        tracks_by_class: Dict[int, List[int]] = defaultdict(list)
        dets_by_class: Dict[int, List[int]] = defaultdict(list)

        for tid in track_ids:
            tracks_by_class[self.tracks[tid].class_id].append(tid)
        for j, det in enumerate(detections):
            dets_by_class[det["class_id"]].append(j)

        matched_track_ids = []
        matched_det_indices = []
        unmatched_track_ids_set = set(track_ids)
        unmatched_det_indices_set = set(range(len(detections)))

        # Run Hungarian Algorithm separately per class
        for class_id, cls_track_ids in tracks_by_class.items():
            cls_det_indices = dets_by_class.get(class_id, [])
            if not cls_det_indices:
                continue  # no detections for this class — tracks stay unmatched

            cls_detections = [detections[j] for j in cls_det_indices]
            dist_matrix = self._compute_distance_matrix(cls_detections, cls_track_ids)

            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] <= self.max_distance:
                    tid = cls_track_ids[r]
                    det_idx = cls_det_indices[c]
                    matched_track_ids.append(tid)
                    matched_det_indices.append(det_idx)
                    unmatched_track_ids_set.discard(tid)
                    unmatched_det_indices_set.discard(det_idx)

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
        num_tracks = len(track_ids)
        num_detections = len(detections)

        distance_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]

            # Predicted measurement: where does this track expect to see an object?
            # get_predicted_measurement() applies the measurement matrix H to the
            # current state s = [x, y, vx, vy] and returns the expected [x, y].
            z_pred = track.get_predicted_measurement()

            for j, detection in enumerate(detections):
                # All detections passed here belong to the same class as this track
                # (guaranteed by the per-class grouping in _associate).
                # No class_id check needed here.

                # Actual measurement: detector position [x, y] in mm
                z = np.array(
                    [detection["center"]["x"], detection["center"]["y"]],
                    dtype=np.float32,
                )

                # Distance-dependent innovation covariance S = H @ Σ @ H^T + R(d)
                d = np.linalg.norm(z)
                R_dynamic = track.get_R_at_distance(d)
                S = track.covariance[0:2, 0:2] + R_dynamic

                # Innovation (residual): difference between actual and predicted measurement
                # y = z - ẑ   (how far off was the prediction?)
                y = z - z_pred

                # Mahalanobis distance: d = sqrt(yᵀ S⁻¹ y)
                # Unlike Euclidean distance, Mahalanobis normalizes by the uncertainty S.
                # A large residual y that lies within the uncertainty ellipse of S will
                # still result in a small Mahalanobis distance — correctly matching a
                # track that has high uncertainty (e.g. just after creation or occlusion).
                # The threshold self.max_distance (default 3.03) corresponds to the
                # 99% confidence region of a 2D chi-squared distribution.
                try:
                    distance_squared = y.T @ np.linalg.inv(S) @ y
                    # Clamp to zero before sqrt: floating-point errors in inv(S) can
                    # produce tiny negative values, causing sqrt to return nan silently.
                    distance = np.sqrt(max(0.0, float(distance_squared)))
                except np.linalg.LinAlgError:
                    # S is singular (numerically degenerate covariance) — fall back to
                    # plain Euclidean distance as a safe approximation
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
            r_dist_ref=self.r_dist_ref,
            sigma_pos_init=self.sigma_pos_init,
            sigma_vel_init=self.sigma_vel_init,
        )
        # self.tracks.append(new_track)
        # self.total_tracks_created += 1
        # Neu: Speichern im Dictionary unter der ID
        self.tracks[track_id] = new_track
        self.total_tracks_created += 1

    def _deduplicate_tracks(self) -> None:
        """
        Remove duplicate tracks of the same class that are too close together.

        Track-level NMS (Non-Maximum Suppression): for each pair of tracks with
        the same class_id, if their Euclidean distance is below duplicate_distance,
        the weaker track is deleted. "Stronger" = more hits; tie-break = lower
        position uncertainty.
        """
        ids_to_delete: Set[int] = set()
        track_ids = list(self.tracks.keys())

        for i in range(len(track_ids)):
            id_a = track_ids[i]
            if id_a in ids_to_delete:
                continue
            track_a = self.tracks[id_a]

            for j in range(i + 1, len(track_ids)):
                id_b = track_ids[j]
                if id_b in ids_to_delete:
                    continue
                track_b = self.tracks[id_b]

                # Only compare tracks of the same class
                if track_a.class_id != track_b.class_id:
                    continue

                # Euclidean distance between positions
                dx = track_a.state[0] - track_b.state[0]
                dy = track_a.state[1] - track_b.state[1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < self.duplicate_distance:
                    # Keep the stronger track (more hits, lower uncertainty as tie-break)
                    if track_a.hits > track_b.hits:
                        loser = id_b
                    elif track_b.hits > track_a.hits:
                        loser = id_a
                    else:
                        # Tie-break: lower position uncertainty wins
                        if (
                            track_a.get_position_uncertainty()
                            <= track_b.get_position_uncertainty()
                        ):
                            loser = id_b
                        else:
                            loser = id_a

                    ids_to_delete.add(loser)
                    # If track_a was the loser, stop comparing it
                    if loser == id_a:
                        break

        for track_id in ids_to_delete:
            del self.tracks[track_id]
            self.total_tracks_deleted += 1

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

    def get_timing_log(self) -> List[Dict]:
        """
        Get the full timing log for post-run analysis.

        Returns a list of dictionaries, one per frame, each containing:
            - frame: Frame number
            - predict_us, associate_us, update_us, create_delete_us, total_us: Timings in µs
            - num_tracks: Number of active tracks after this frame
            - num_detections: Number of detections in this frame
        """
        return self._timing_log

    def reset(self, reset_id_counter: bool = True) -> None:
        """
        Reset the tracker (delete all tracks and reset statistics).

        Args:
            reset_id_counter: If True, reset the ID counter to id_offset (default: True)
                             Set to False if you want to preserve continuous ID numbering
        """
        self.tracks = {}
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

        # Reset instance-level ID counter (not global!)
        if reset_id_counter:
            # Reset to initial offset (preserves separation between trackers)
            self._next_track_id = self._initial_id_offset
