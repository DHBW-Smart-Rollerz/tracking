"""
Multi-Topic Object Tracking Node for tracking both objects and signs.

This node maintains three separate trackers:
- One for moving objects (cars, pedestrians) from /object_detection/object
- One for static signs from /object_detection/sign
- One for crossing lane lines from /crossing_detection/result

It bundles all confirmed tracks into a single state_msgs/State message
and publishes them at a fixed frequency.
"""

import csv
import os
import time


import rclpy
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from std_msgs.msg import Float32MultiArray

import std_msgs
import std_msgs.msg
import state_msgs.msg

from tracking.tracker import MultiObjectTracker


_TRACKER_PARAM_KEYS = [
    "max_age",
    "min_hits",
    "min_age",
    "max_distance",
    "q_pos",
    "q_vel",
    "r_pos",
    "sigma_pos_init",
    "sigma_vel_init",
]


class ObjectTrackingNode(SmartyNode):
    """ROS2 Object Tracking Node with separate trackers for objects, signs, and crossings."""

    def __init__(self):
        """Initialize the ObjectTrackingNode."""
        super().__init__(
            "object_tracking_node",
            "tracking",
            node_parameters={
                # Subscriber topics
                "image_subscriber": None,
                "object_detection_subscriber": None,
                "sign_detection__subscriber": None,
                "crossing_detection_subscriber": None,
                # Publisher topics (nur noch einer!)
                "state_publisher": None,
                # Node settings
                "state": None,
                "debug": None,
                "export_timing_csv": None,
                # Common parameters
                "dt": None,
                "publish_interval_ms": None,  # Neuer Parameter für den Timer
                "max_x": None,
                "max_y": None,
                # Object tracker parameters
                "object_max_age": None,
                "object_min_hits": None,
                "object_min_age": None,
                "object_max_distance": None,
                "object_q_pos": None,
                "object_q_vel": None,
                "object_r_pos": None,
                "object_sigma_pos_init": None,
                "object_sigma_vel_init": None,
                # Sign tracker parameters
                "sign_max_age": None,
                "sign_min_hits": None,
                "sign_min_age": None,
                "sign_max_distance": None,
                "sign_q_pos": None,
                "sign_q_vel": None,
                "sign_r_pos": None,
                "sign_sigma_pos_init": None,
                "sign_sigma_vel_init": None,
                # Crossing tracker parameters
                "crossing_max_age": None,
                "crossing_min_hits": None,
                "crossing_min_age": None,
                "crossing_max_distance": None,
                "crossing_q_pos": None,
                "crossing_q_vel": None,
                "crossing_r_pos": None,
                "crossing_sigma_pos_init": None,
                "crossing_sigma_vel_init": None,
            },
            subscribed_topics={
                "object_detection_subscriber": (
                    Float32MultiArray,
                    self.object_detection_callback,
                    1,
                ),
                "sign_detection__subscriber": (
                    Float32MultiArray,
                    self.sign_detection_callback,
                    1,
                ),
                "crossing_detection_subscriber": (
                    Float32MultiArray,
                    self.crossing_detection_callback,
                    1,
                ),
            },
            published_topics={
                # Gebündelter State-Publisher mit deiner Custom Message
                "state_publisher": (state_msgs.msg.State, 1),
            },
        )

        # Initialize three separate trackers
        self.object_tracker = self._create_tracker("object", id_offset=0)
        self.sign_tracker = self._create_tracker("sign", id_offset=10000)
        self.crossing_tracker = self._create_tracker("crossing", id_offset=20000)

        # Timestamp tracking for dynamic dt calculation
        self.last_object_time = None
        self.last_sign_time = None
        self.last_crossing_time = None

        self._log_startup_info()

        # Timer für das Veröffentlichen mit fester Frequenz einrichten
        interval_sec = self._param("publish_interval_ms") / 1000.0
        self.publish_timer = self.create_timer(
            interval_sec, self.publish_state_callback
        )
        self.get_logger().info(
            f"State publisher timer set to {self._param('publish_interval_ms')} ms ({1/interval_sec:.1f} Hz)"
        )

    # -------------------------------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------------------------------

    def _param(self, name: str):
        return self.get_parameter(name).value

    def _get_tracker_params(self, prefix: str) -> dict:
        params = {key: self._param(f"{prefix}_{key}") for key in _TRACKER_PARAM_KEYS}
        params["max_x"] = self._param("max_x")
        params["max_y"] = self._param("max_y")
        return params

    def _create_tracker(self, prefix: str, id_offset: int) -> MultiObjectTracker:
        params = self._get_tracker_params(prefix)
        return MultiObjectTracker(**params, id_offset=id_offset)

    # -------------------------------------------------------------------------
    # Detection callbacks (Nur noch für das Update zuständig!)
    # -------------------------------------------------------------------------

    def _calculate_dt(self, last_time, current_time) -> float:
        if last_time is not None:
            dt = (current_time - last_time).nanoseconds * 1e-9
            return max(0.01, min(1.0, dt))
        return self._param("dt")

    def _process_detection(self, msg, tracker, tracker_name, last_time):
        """Shared detection processing logic for updating trackers."""
        t_start = time.perf_counter()

        current_time = self.get_clock().now()
        dt = self._calculate_dt(last_time, current_time)

        detections = self.parse_detections(msg)

        # Tracker aktualisieren (gibt confirmed_tracks zurück, aber wir ignorieren
        # den Return-Wert hier, da der Timer sie asynchron abholt)
        tracker.process_step(detections, dt=dt)

        if self._debug:
            stats = tracker.get_statistics()
            self.get_logger().info(
                f"{tracker_name} stats: active={stats['active_tracks']}, "
                f"confirmed={stats['confirmed_tracks']}, "
                f"frame={stats['frame_count']} | dt={dt:.4f}s"
            )

        return current_time

    def object_detection_callback(self, msg: Float32MultiArray):
        self.last_object_time = self._process_detection(
            msg, self.object_tracker, "object", self.last_object_time
        )

    def sign_detection_callback(self, msg: Float32MultiArray):
        self.last_sign_time = self._process_detection(
            msg, self.sign_tracker, "sign", self.last_sign_time
        )

    def crossing_detection_callback(self, msg: Float32MultiArray):
        self.last_crossing_time = self._process_detection(
            msg, self.crossing_tracker, "crossing", self.last_crossing_time
        )

    # -------------------------------------------------------------------------
    # Publisher Callback (Timer-basiert)
    # -------------------------------------------------------------------------

    def publish_state_callback(self):
        """Called by the timer to publish the combined state of all trackers."""
        if not self._param("state") == NodeState.ACTIVE.value:
            return

        # 1. State Message vorbereiten
        state_msg = state_msgs.msg.State()

        # 2. Confirmed Tracks von allen drei Trackern einsammeln
        all_confirmed_tracks = []
        all_confirmed_tracks.extend(self.object_tracker.get_confirmed_tracks())
        all_confirmed_tracks.extend(self.sign_tracker.get_confirmed_tracks())
        all_confirmed_tracks.extend(self.crossing_tracker.get_confirmed_tracks())

        # 3. Dictionaries in TrackedObject.msg umwandeln
        for obj in all_confirmed_tracks:
            tracked_obj = state_msgs.msg.TrackedObject()

            # Typkonvertierung zu float64, wie in TrackedObject.msg gefordert
            tracked_obj.tracked_id = float(obj["track_id"])
            tracked_obj.class_id = float(obj["class_id"])
            tracked_obj.position_x = float(obj["position"]["x"])
            tracked_obj.position_y = float(obj["position"]["y"])
            tracked_obj.velocity_x = float(obj["velocity"]["vx"])
            tracked_obj.velocity_y = float(obj["velocity"]["vy"])
            tracked_obj.confidence = float(obj["confidence"])
            tracked_obj.width = float(obj["width"])

            # Dem Array in der State-Message hinzufügen
            state_msg.tracked_objects.append(tracked_obj)

        # 4. Senden
        self.state_publisher.publish(state_msg)

    # -------------------------------------------------------------------------
    # Message parsing
    # -------------------------------------------------------------------------

    def parse_detections(self, msg: Float32MultiArray):
        """Parse the Float32MultiArray into individual detections."""
        data = msg.data
        values_per_detection = 6

        if len(data) == 0:
            return []

        if len(data) % values_per_detection != 0:
            self.get_logger().warn(
                f"Unexpected data length: {len(data)} (not divisible by {values_per_detection})"
            )
            return []

        detections = []
        for i in range(0, len(data), values_per_detection):
            bl_x, bl_y = data[i + 1], data[i + 2]
            br_x, br_y = data[i + 3], data[i + 4]

            detections.append(
                {
                    "class_id": int(data[i]),
                    "score": data[i + 5],
                    "center": {"x": (bl_x + br_x) / 2.0, "y": (bl_y + br_y) / 2.0},
                    "width": abs(br_x - bl_x),
                    "bottom_left": {"x": bl_x, "y": bl_y},
                    "bottom_right": {"x": br_x, "y": br_y},
                }
            )

        return detections

    # -------------------------------------------------------------------------
    # Performance data export
    # -------------------------------------------------------------------------

    def _export_timing_data(self, filepath: str = None) -> str:
        """
        Export timing data from all trackers to a single CSV file.

        Args:
            filepath: Output file path. Defaults to ~/tracking_timing_<timestamp>.csv

        Returns:
            The file path where data was written.
        """
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Absoluter Pfad zu deinem Source-Ordner
            base_dir = os.path.expanduser(
                "~/smarty_workspace/src/tracking/performance_measurements"
            )

            # Sicherstellen, dass der Ordner existiert (falls er mal gelöscht wird)
            os.makedirs(base_dir, exist_ok=True)

            filepath = os.path.join(base_dir, f"tracking_timing_{timestamp}.csv")

        fieldnames = [
            "frame",
            "tracker",
            "timestamp",
            "predict_us",
            "associate_us",
            "update_us",
            "create_delete_us",
            "total_us",
            "num_tracks",
            "num_detections",
        ]

        total_rows = 0
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tracker_name, tracker in [
                ("object", self.object_tracker),
                ("sign", self.sign_tracker),
                ("crossing", self.crossing_tracker),
            ]:
                for entry in tracker.get_timing_log():
                    row = {"tracker": tracker_name}
                    row.update(entry)
                    writer.writerow(row)
                    total_rows += 1

        self.get_logger().info(f"Timing data exported: {total_rows} rows -> {filepath}")
        return filepath

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log_startup_info(self):
        self.get_logger().info(
            "ObjectTrackingNode initialized with 3 trackers (Timer-based publishing)"
        )
        self.get_logger().info("Subscribed topics:")
        for key in self.subscribed_topics:
            self.get_logger().info(f"  {key}: {self._param(key)}")
        self.get_logger().info("Published topics:")
        for key in self.published_topics:
            self.get_logger().info(f"  {key}: {self._param(key)}")

    def _log_tracker_stats(self, name: str, stats: dict):
        self.get_logger().info(f"{name} Tracker Statistics:")
        self.get_logger().info(f"  Frames processed: {stats['frame_count']}")
        self.get_logger().info(f"  Tracks created:   {stats['total_created']}")
        self.get_logger().info(f"  Tracks deleted:   {stats['total_deleted']}")
        self.get_logger().info(f"  Active tracks:    {stats['active_tracks']}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("=" * 60)
        node.get_logger().info("Shutting down Object Tracking Node")
        node.get_logger().info("=" * 60)

        # Export timing data to CSV before shutdown
        try:
            if node._param("export_timing_csv"):
                csv_path = node._export_timing_data()
                node.get_logger().info(f"Timing CSV saved to: {csv_path}")
            else:
                node.get_logger().info(
                    "Timing CSV export disabled (export_timing_csv: false)"
                )
        except Exception as e:
            node.get_logger().error(f"Failed to export timing data: {e}")

        for name, tracker in [
            ("Object", node.object_tracker),
            ("Sign", node.sign_tracker),
            ("Crossing", node.crossing_tracker),
        ]:
            node._log_tracker_stats(name, tracker.get_statistics())
            node.get_logger().info("-" * 60)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
