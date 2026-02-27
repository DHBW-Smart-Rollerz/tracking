"""
Multi-Topic Object Tracking Node for tracking both objects and signs.

This node maintains three separate trackers:
- One for moving objects (cars, pedestrians) from /object_detection/object
- One for static signs from /object_detection/sign
- One for crossing lane lines from /crossing_detection/result

Each tracker has its own optimized parameters.

Parameter flow (single source of truth):
  tracking_params.yaml → node_parameters (declaration defaults) → MultiObjectTracker
  The YAML file is the authoritative source.
"""

import rclpy
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from std_msgs.msg import Float32MultiArray

from tracking.tracker import MultiObjectTracker


# Tracker parameter names (shared across all tracker types).
# These get prefixed with "object_", "sign_", or "crossing_" in the YAML/node.
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
                # All parameters here are for initialization ONLY. The values are set in tracking_params.yaml
                # Please don't fill any values in here to prevent confusion.
                # Subscriber topics
                "image_subscriber": None,
                "object_detection_subscriber": None,
                "sign_detection__subscriber": None,
                "crossing_detection_subscriber": None,
                # Publisher topics
                "object_tracking_publisher": None,
                "sign_tracking_publisher": None,
                "crossing_tracking_publisher": None,
                # Node settings
                "state": None,
                "debug": None,
                # Common parameters
                "dt": None,
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
                "object_tracking_publisher": (Float32MultiArray, 1),
                "sign_tracking_publisher": (Float32MultiArray, 1),
                "crossing_tracking_publisher": (Float32MultiArray, 1),
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

    # -------------------------------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------------------------------

    def _param(self, name: str):
        """Read a single ROS parameter value."""
        return self.get_parameter(name).value

    def _get_tracker_params(self, prefix: str) -> dict:
        """
        Read all tracker parameters for a given prefix from the ROS parameter server.

        Args:
            prefix: One of "object", "sign", "crossing".

        Returns:
            Dict with keys matching MultiObjectTracker.__init__ kwargs.
        """
        params = {key: self._param(f"{prefix}_{key}") for key in _TRACKER_PARAM_KEYS}
        params["max_x"] = self._param("max_x")
        params["max_y"] = self._param("max_y")
        return params

    def _create_tracker(self, prefix: str, id_offset: int) -> MultiObjectTracker:
        """
        Create a MultiObjectTracker from ROS parameters.

        Args:
            prefix: Parameter prefix ("object", "sign", or "crossing").
            id_offset: Starting ID for this tracker (0, 10000, or 20000).

        Returns:
            Configured MultiObjectTracker instance.
        """
        params = self._get_tracker_params(prefix)
        return MultiObjectTracker(**params, id_offset=id_offset)

    # -------------------------------------------------------------------------
    # Detection callbacks
    # -------------------------------------------------------------------------

    def _calculate_dt(self, last_time, current_time) -> float:
        """Calculate dynamic time step from timestamps, with clamping."""
        if last_time is not None:
            dt = (current_time - last_time).nanoseconds * 1e-9
            return max(0.01, min(1.0, dt))  # Clamp to [10ms, 1s]
        return self._param("dt")  # Fallback for first frame

    def _process_detection(self, msg, tracker, tracker_name, publisher, last_time):
        """
        Shared detection processing logic for all three tracker types.

        Args:
            msg: Float32MultiArray containing detection data.
            tracker: The MultiObjectTracker instance.
            tracker_name: Name for logging ("object", "sign", "crossing").
            publisher: The ROS publisher for this tracker type.
            last_time: Previous callback timestamp (or None).

        Returns:
            Current timestamp (to store as last_time for next call).
        """
        current_time = self.get_clock().now()
        dt = self._calculate_dt(last_time, current_time)

        if self._debug:
            self.get_logger().info("=" * 60)
            self.get_logger().info(
                f"Received {tracker_name.upper()} detection | "
                f"len={len(msg.data)} | dt={dt:.4f}s ({1/dt:.1f} Hz)"
            )

        detections = self.parse_detections(msg)

        if self._debug and detections:
            for i, det in enumerate(detections):
                self.get_logger().info(
                    f"  Det {i}: class={det['class_id']}, "
                    f"pos=({det['center']['x']:.1f}, {det['center']['y']:.1f}), "
                    f"score={det['score']:.3f}"
                )

        confirmed_tracks = tracker.update(detections, dt=dt)

        if self._debug:
            stats = tracker.get_statistics()
            self.get_logger().info(
                f"{tracker_name} stats: active={stats['active_tracks']}, "
                f"confirmed={stats['confirmed_tracks']}, "
                f"frame={stats['frame_count']}"
            )

        if confirmed_tracks:
            tracking_msg = self.create_tracking_message(confirmed_tracks)
            publisher.publish(tracking_msg)

            if self._debug:
                self.get_logger().info(
                    f"Published {len(confirmed_tracks)} {tracker_name} track(s)"
                )
                for track in confirmed_tracks:
                    self.get_logger().info(
                        f"  ID {track['track_id']}: class={track['class_id']}, "
                        f"pos=({track['position']['x']:.1f}, {track['position']['y']:.1f}), "
                        f"vel=({track['velocity']['vx']:.1f}, {track['velocity']['vy']:.1f}), "
                        f"conf={track['confidence']:.3f}, hits={track['hits']}"
                    )
        elif self._debug:
            self.get_logger().info(f"No confirmed {tracker_name} tracks to publish")

        return current_time

    def object_detection_callback(self, msg: Float32MultiArray):
        """Callback for object detections (cars, pedestrians)."""
        self.last_object_time = self._process_detection(
            msg,
            self.object_tracker,
            "object",
            self.object_tracking_publisher,
            self.last_object_time,
        )

    def sign_detection_callback(self, msg: Float32MultiArray):
        """Callback for sign detections."""
        self.last_sign_time = self._process_detection(
            msg,
            self.sign_tracker,
            "sign",
            self.sign_tracking_publisher,
            self.last_sign_time,
        )

    def crossing_detection_callback(self, msg: Float32MultiArray):
        """Callback for crossing detections (ego/opp lane lines)."""
        self.last_crossing_time = self._process_detection(
            msg,
            self.crossing_tracker,
            "crossing",
            self.crossing_tracking_publisher,
            self.last_crossing_time,
        )

    # -------------------------------------------------------------------------
    # Message parsing / creation
    # -------------------------------------------------------------------------

    def parse_detections(self, msg: Float32MultiArray):
        """
        Parse the Float32MultiArray into individual detections.

        Format per detection (6 values):
        [class_id, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, score]

        Coordinates are in ego-frame millimeters.
        """
        data = msg.data
        values_per_detection = 6

        if len(data) == 0:
            if self._debug:
                self.get_logger().info("No detections in this frame")
            return []

        if len(data) % values_per_detection != 0:
            self.get_logger().warn(
                f"Unexpected data length: {len(data)} "
                f"(not divisible by {values_per_detection})"
            )
            self.get_logger().info(f"Raw data: {list(data)}")
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

    def create_tracking_message(self, tracked_objects):
        """
        Create a Float32MultiArray from tracked objects.

        Format per object: [track_id, class_id, x, y, vx, vy, confidence, width]
        """
        msg = Float32MultiArray()
        data = []

        for obj in tracked_objects:
            data.extend(
                [
                    float(obj["track_id"]),
                    float(obj["class_id"]),
                    float(obj["position"]["x"]),
                    float(obj["position"]["y"]),
                    float(obj["velocity"]["vx"]),
                    float(obj["velocity"]["vy"]),
                    float(obj["confidence"]),
                    float(obj["width"]),
                ]
            )

        msg.data = data
        return msg

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log_startup_info(self):
        """Log subscription and publication info at startup."""
        self.get_logger().info("ObjectTrackingNode initialized with 3 trackers")
        self.get_logger().info("Subscribed topics:")
        for key in self.subscribed_topics:
            self.get_logger().info(f"  {key}: {self._param(key)}")
        self.get_logger().info("Published topics:")
        for key in self.published_topics:
            self.get_logger().info(f"  {key}: {self._param(key)}")

    def _log_tracker_stats(self, name: str, stats: dict):
        """Log statistics for a single tracker."""
        self.get_logger().info(f"{name} Tracker Statistics:")
        self.get_logger().info(f"  Frames processed: {stats['frame_count']}")
        self.get_logger().info(f"  Tracks created:   {stats['total_created']}")
        self.get_logger().info(f"  Tracks deleted:   {stats['total_deleted']}")
        self.get_logger().info(f"  Active tracks:    {stats['active_tracks']}")


def main(args=None):
    """Main function to start the ObjectTrackingNode."""
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

        for name, tracker in [
            ("Object", node.object_tracker),
            ("Sign", node.sign_tracker),
            ("Crossing", node.crossing_tracker),
        ]:
            node._log_tracker_stats(name, tracker.get_statistics())
            node.get_logger().info("-" * 60)

        node.get_logger().info("=" * 60)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
