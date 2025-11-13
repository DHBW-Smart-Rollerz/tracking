"""
Multi-Topic Object Tracking Node for tracking both objects and signs.

This node maintains two separate trackers:
- One for moving objects (cars, pedestrians) from /object_detection/object
- One for static signs from /object_detection/sign

Each tracker has its own optimized parameters.
"""

import rclpy
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from std_msgs.msg import Float32MultiArray

from tracking.tracker import MultiObjectTracker


class ObjectTrackingNode(SmartyNode):
    """ROS2 Object Tracking Node with separate trackers for objects and signs."""

    def __init__(self):
        """Initialize the ObjectTrackingNode."""
        super().__init__(
            "object_tracking_node",
            "tracking",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/image/undistorted",
                "object_detection_subscriber": "/object_detection/object",
                "sign_detection__subscriber": "/object_detection/sign",
                # Publisher topics
                "object_tracking_publisher": "/object_tracking/tracked_objects",
                "sign_tracking_publisher": "/sign_tracking/tracked_signs",
                # Parameters
                "state": NodeState.ACTIVE.value,
                "debug": False,
                # Common parameters
                "dt": 0.1,  # Time step in seconds (10 Hz - only used as fallback)
                "max_x": 5000.0,  # Max valid x position (mm)
                "max_y": 3000.0,  # Max valid y position (mm)
                # Object tracker parameters (moving objects)
                "object_max_age": 5,  # Max frames without update
                "object_min_hits": 3,  # Min hits for confirmation
                "object_min_age": 3,  # Min age for confirmation
                "object_max_distance": 9.21,  # Max Mahalanobis distance (chi-squared, 99% confidence for 2D)
                "object_q_pos": 50.0,  # Process noise: position (mm)
                "object_q_vel": 100.0,  # Process noise: velocity (mm/s)
                "object_r_pos": 100.0,  # Measurement noise: position (mm)
                "object_sigma_pos_init": 500.0,  # Initial position uncertainty (mm)
                "object_sigma_vel_init": 1000.0,  # Initial velocity uncertainty (mm/s)
                # Sign tracker parameters (static objects)
                "sign_max_age": 10,  # Longer for signs (don't disappear)
                "sign_min_hits": 2,  # Faster confirmation for signs
                "sign_min_age": 2,  # Shorter confirmation time
                "sign_max_distance": 9.21,  # Max Mahalanobis distance (chi-squared, 99% confidence for 2D)
                "sign_q_pos": 25.0,  # Lower process noise (static)
                "sign_q_vel": 50.0,  # Lower velocity noise
                "sign_r_pos": 100.0,  # Measurement noise: position (mm)
                "sign_sigma_pos_init": 300.0,  # Lower initial position uncertainty
                "sign_sigma_vel_init": 500.0,  # Lower initial velocity uncertainty
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
            },
            published_topics={
                "object_tracking_publisher": (Float32MultiArray, 1),
                "sign_tracking_publisher": (Float32MultiArray, 1),
            },
        )

        # Initialize two separate trackers with different parameters
        self.object_tracker = self._create_object_tracker()
        self.sign_tracker = self._create_sign_tracker()

        # Timestamp tracking for dynamic dt calculation
        self.last_object_time = None
        self.last_sign_time = None

        self.get_logger().info("ObjectTrackingNode initialized with dual trackers")
        self.get_logger().info("Subscribed topics:")
        for key, (msg_type, callback, queue_size) in self.subscribed_topics.items():
            self.get_logger().info(f"  {key}: {self.get_parameter(key).value}")
        self.get_logger().info("Published topics:")
        for key, (msg_type, queue_size) in self.published_topics.items():
            self.get_logger().info(f"  {key}: {self.get_parameter(key).value}")

    def _create_object_tracker(self):
        """
        Create tracker for moving objects (cars, pedestrians).

        Uses standard parameters optimized for dynamic objects.

        Returns:
            MultiObjectTracker instance for objects
        """
        return MultiObjectTracker(
            max_age=self.object_max_age,
            min_hits=self.object_min_hits,
            min_age=self.object_min_age,
            max_distance=self.object_max_distance,
            max_x=self.max_x,
            max_y=self.max_y,
            q_pos=self.object_q_pos,
            q_vel=self.object_q_vel,
            r_pos=self.object_r_pos,
            sigma_pos_init=self.object_sigma_pos_init,
            sigma_vel_init=self.object_sigma_vel_init,
        )

    def _create_sign_tracker(self):
        """
        Create tracker for static signs.

        Uses optimized parameters for static objects:
        - Longer max_age (signs don't disappear quickly)
        - Fewer min_hits (faster confirmation)
        - Lower process noise (signs don't move)

        Returns:
            MultiObjectTracker instance for signs
        """
        return MultiObjectTracker(
            max_age=self.sign_max_age,
            min_hits=self.sign_min_hits,
            min_age=self.sign_min_age,
            max_distance=self.sign_max_distance,
            max_x=self.max_x,
            max_y=self.max_y,
            q_pos=self.sign_q_pos,
            q_vel=self.sign_q_vel,
            r_pos=self.sign_r_pos,
            sigma_pos_init=self.sign_sigma_pos_init,
            sigma_vel_init=self.sign_sigma_vel_init,
        )

    @property
    def dt(self) -> float:
        """Return the time step parameter."""
        return self.get_parameter("dt").value  # type: ignore

    @property
    def max_x(self) -> float:
        """Return the max_x parameter."""
        return self.get_parameter("max_x").value  # type: ignore

    @property
    def max_y(self) -> float:
        """Return the max_y parameter."""
        return self.get_parameter("max_y").value  # type: ignore

    @property
    def object_max_age(self) -> int:
        """Return the object_max_age parameter."""
        return self.get_parameter("object_max_age").value  # type: ignore

    @property
    def object_min_hits(self) -> int:
        """Return the object_min_hits parameter."""
        return self.get_parameter("object_min_hits").value  # type: ignore

    @property
    def object_min_age(self) -> int:
        """Return the object_min_age parameter."""
        return self.get_parameter("object_min_age").value  # type: ignore

    @property
    def object_max_distance(self) -> float:
        """Return the object_max_distance parameter."""
        return self.get_parameter("object_max_distance").value  # type: ignore

    @property
    def object_q_pos(self) -> float:
        """Return the object_q_pos parameter."""
        return self.get_parameter("object_q_pos").value  # type: ignore

    @property
    def object_q_vel(self) -> float:
        """Return the object_q_vel parameter."""
        return self.get_parameter("object_q_vel").value  # type: ignore

    @property
    def object_r_pos(self) -> float:
        """Return the object_r_pos parameter."""
        return self.get_parameter("object_r_pos").value  # type: ignore

    @property
    def object_sigma_pos_init(self) -> float:
        """Return the object_sigma_pos_init parameter."""
        return self.get_parameter("object_sigma_pos_init").value  # type: ignore

    @property
    def object_sigma_vel_init(self) -> float:
        """Return the object_sigma_vel_init parameter."""
        return self.get_parameter("object_sigma_vel_init").value  # type: ignore

    @property
    def sign_max_age(self) -> int:
        """Return the sign_max_age parameter."""
        return self.get_parameter("sign_max_age").value  # type: ignore

    @property
    def sign_min_hits(self) -> int:
        """Return the sign_min_hits parameter."""
        return self.get_parameter("sign_min_hits").value  # type: ignore

    @property
    def sign_min_age(self) -> int:
        """Return the sign_min_age parameter."""
        return self.get_parameter("sign_min_age").value  # type: ignore

    @property
    def sign_max_distance(self) -> float:
        """Return the sign_max_distance parameter."""
        return self.get_parameter("sign_max_distance").value  # type: ignore

    @property
    def sign_q_pos(self) -> float:
        """Return the sign_q_pos parameter."""
        return self.get_parameter("sign_q_pos").value  # type: ignore

    @property
    def sign_q_vel(self) -> float:
        """Return the sign_q_vel parameter."""
        return self.get_parameter("sign_q_vel").value  # type: ignore

    @property
    def sign_r_pos(self) -> float:
        """Return the sign_r_pos parameter."""
        return self.get_parameter("sign_r_pos").value  # type: ignore

    @property
    def sign_sigma_pos_init(self) -> float:
        """Return the sign_sigma_pos_init parameter."""
        return self.get_parameter("sign_sigma_pos_init").value  # type: ignore

    @property
    def sign_sigma_vel_init(self) -> float:
        """Return the sign_sigma_vel_init parameter."""
        return self.get_parameter("sign_sigma_vel_init").value  # type: ignore

    def object_detection_callback(self, msg: Float32MultiArray):
        """
        Callback for object detections (cars, pedestrians).

        Args:
            msg: Float32MultiArray containing detection data
        """
        # Calculate dynamic dt based on actual timestamps
        current_time = self.get_clock().now()
        if self.last_object_time is not None:
            dt = (current_time - self.last_object_time).nanoseconds * 1e-9
            # Clamp dt to reasonable range (10ms to 1s)
            dt = max(0.01, min(1.0, dt))
        else:
            # First frame: use default dt
            dt = self.dt
        
        self.last_object_time = current_time

        if self._debug:
            self.get_logger().info("=" * 60)
            self.get_logger().info("Received OBJECT detection message!")
            self.get_logger().info(f"Data length: {len(msg.data)}")
            self.get_logger().info(f"Time step dt: {dt:.4f}s ({1/dt:.1f} Hz)")

        # Parse the detection data
        detections = self.parse_detections(msg)

        if self._debug and detections:
            self.get_logger().info(f"Parsed {len(detections)} object detection(s):")
            for i, det in enumerate(detections):
                self.get_logger().info(
                    f"  Detection {i}: class={det['class_id']}, "
                    f"pos=({det['center']['x']:.1f}, {det['center']['y']:.1f}), "
                    f"score={det['score']:.3f}"
                )

        # Update tracker with detections and dynamic dt
        confirmed_tracks = self.object_tracker.update(detections, dt=dt)

        if self._debug:
            # Log tracker statistics
            stats = self.object_tracker.get_statistics()
            self.get_logger().info(
                f"Object tracker stats: active={stats['active_tracks']}, "
                f"confirmed={stats['confirmed_tracks']}, "
                f"frame={stats['frame_count']}"
            )

        # Publish confirmed tracks
        if confirmed_tracks:
            tracking_msg = self.create_tracking_message(confirmed_tracks)
            self.object_tracking_publisher.publish(tracking_msg)  # type: ignore

            if self._debug:
                self.get_logger().info(
                    f"Published {len(confirmed_tracks)} confirmed object track(s):"
                )
                for track in confirmed_tracks:
                    self.get_logger().info(
                        f"  Track ID {track['track_id']}: class={track['class_id']}, "
                        f"pos=({track['position']['x']:.1f}, {track['position']['y']:.1f}), "
                        f"vel=({track['velocity']['vx']:.1f}, {track['velocity']['vy']:.1f}), "
                        f"confidence={track['confidence']:.3f}, hits={track['hits']}"
                    )
        elif self._debug:
            self.get_logger().info("No confirmed object tracks to publish")

    def sign_detection_callback(self, msg: Float32MultiArray):
        """
        Callback for sign detections.

        Args:
            msg: Float32MultiArray containing detection data
        """
        # Calculate dynamic dt based on actual timestamps
        current_time = self.get_clock().now()
        if self.last_sign_time is not None:
            dt = (current_time - self.last_sign_time).nanoseconds * 1e-9
            # Clamp dt to reasonable range (10ms to 1s)
            dt = max(0.01, min(1.0, dt))
        else:
            # First frame: use default dt
            dt = self.dt
        
        self.last_sign_time = current_time

        if self._debug:
            self.get_logger().info("=" * 60)
            self.get_logger().info("Received SIGN detection message!")
            self.get_logger().info(f"Data length: {len(msg.data)}")
            self.get_logger().info(f"Time step dt: {dt:.4f}s ({1/dt:.1f} Hz)")

        # Parse the detection data
        detections = self.parse_detections(msg)

        if self._debug and detections:
            self.get_logger().info(f"Parsed {len(detections)} sign detection(s):")
            for i, det in enumerate(detections):
                self.get_logger().info(
                    f"  Detection {i}: class={det['class_id']}, "
                    f"pos=({det['center']['x']:.1f}, {det['center']['y']:.1f}), "
                    f"score={det['score']:.3f}"
                )

        # Update tracker with detections and dynamic dt
        confirmed_tracks = self.sign_tracker.update(detections, dt=dt)

        if self._debug:
            # Log tracker statistics
            stats = self.sign_tracker.get_statistics()
            self.get_logger().info(
                f"Sign tracker stats: active={stats['active_tracks']}, "
                f"confirmed={stats['confirmed_tracks']}, "
                f"frame={stats['frame_count']}"
            )

        # Publish confirmed tracks
        if confirmed_tracks:
            tracking_msg = self.create_tracking_message(confirmed_tracks)
            self.sign_tracking_publisher.publish(tracking_msg)  # type: ignore

            if self._debug:
                self.get_logger().info(
                    f"Published {len(confirmed_tracks)} confirmed sign track(s):"
                )
                for track in confirmed_tracks:
                    self.get_logger().info(
                        f"  Track ID {track['track_id']}: class={track['class_id']}, "
                        f"pos=({track['position']['x']:.1f}, {track['position']['y']:.1f}), "
                        f"vel=({track['velocity']['vx']:.1f}, {track['velocity']['vy']:.1f}), "
                        f"confidence={track['confidence']:.3f}, hits={track['hits']}"
                    )
        elif self._debug:
            self.get_logger().info("No confirmed sign tracks to publish")

    def parse_detections(self, msg: Float32MultiArray):
        """
        Parse the Float32MultiArray into individual detections.

        Format from object_detection_node.py:
        [class_id, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, score]

        Where:
        - class_id: Object class identifier (int)
        - bottom_left_x/y: Bottom-left corner in world coordinates (mm)
        - bottom_right_x/y: Bottom-right corner in world coordinates (mm)
        - score: Detection confidence (0.0 - 1.0)

        Args:
            msg: Float32MultiArray message

        Returns:
            List of detection dictionaries suitable for tracker.update()
        """
        data = msg.data

        if len(data) == 0:
            if self._debug:
                self.get_logger().info("No detections in this frame")
            return []

        # Format: 6 values per detection
        values_per_detection = 6

        if len(data) % values_per_detection != 0:
            self.get_logger().warn(
                f"Unexpected data length: {len(data)} "
                f"(not divisible by {values_per_detection})"
            )
            self.get_logger().info(f"Raw data: {list(data)}")
            return []

        detections = []
        for i in range(0, len(data), values_per_detection):
            # Extract corner points
            bottom_left_x = data[i + 1]
            bottom_left_y = data[i + 2]
            bottom_right_x = data[i + 3]
            bottom_right_y = data[i + 4]

            # Calculate center point (average of two bottom corners)
            center_x = (bottom_left_x + bottom_right_x) / 2.0
            center_y = (bottom_left_y + bottom_right_y) / 2.0

            # Width from distance between left and right corners
            width = abs(bottom_right_x - bottom_left_x)

            # Create detection dictionary for tracker
            detection = {
                "class_id": int(data[i]),
                "score": data[i + 5],
                "center": {"x": center_x, "y": center_y},
                "width": width,
                "bottom_left": {"x": bottom_left_x, "y": bottom_left_y},
                "bottom_right": {"x": bottom_right_x, "y": bottom_right_y},
            }
            detections.append(detection)

        return detections

    def create_tracking_message(self, tracked_objects):
        """
        Create a Float32MultiArray message from tracked objects.

        Message format per tracked object:
        [track_id, class_id, x, y, vx, vy, confidence, width]

        Args:
            tracked_objects: List of tracked object dictionaries from tracker.get_confirmed_tracks()

        Returns:
            Float32MultiArray message
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


def main(args=None):
    """
    Main function to start the ObjectTrackingNode.

    Args:
        args: Launch arguments (default: None)
    """
    rclpy.init(args=args)
    node = ObjectTrackingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Log final statistics before shutdown
        object_stats = node.object_tracker.get_statistics()
        sign_stats = node.sign_tracker.get_statistics()

        node.get_logger().info("=" * 60)
        node.get_logger().info("Shutting down Object Tracking Node")
        node.get_logger().info("=" * 60)
        node.get_logger().info("Object Tracker Statistics:")
        node.get_logger().info(
            f"  Total frames processed: {object_stats['frame_count']}"
        )
        node.get_logger().info(
            f"  Total tracks created: {object_stats['total_created']}"
        )
        node.get_logger().info(
            f"  Total tracks deleted: {object_stats['total_deleted']}"
        )
        node.get_logger().info(f"  Active tracks: {object_stats['active_tracks']}")
        node.get_logger().info("-" * 60)
        node.get_logger().info("Sign Tracker Statistics:")
        node.get_logger().info(f"  Total frames processed: {sign_stats['frame_count']}")
        node.get_logger().info(f"  Total tracks created: {sign_stats['total_created']}")
        node.get_logger().info(f"  Total tracks deleted: {sign_stats['total_deleted']}")
        node.get_logger().info(f"  Active tracks: {sign_stats['active_tracks']}")
        node.get_logger().info("=" * 60)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()