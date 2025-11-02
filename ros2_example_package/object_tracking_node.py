"""Object Tracking Node for tracking detected objects."""

import rclpy
import rclpy.node
from std_msgs.msg import Float32MultiArray
from timing import timer


class ObjectTrackingNode(rclpy.node.Node):
    """ROS2 Object Tracking Node."""

    def __init__(self):
        """Initialize the ObjectTrackingNode."""
        super().__init__("object_tracking_node")

        # Load parameters
        self.load_ros_params()

        # Initialize subscriber
        self.init_subscriber()

        # Initialize tracking variables
        self.tracked_objects = {}  # Dictionary to store tracked objects
        self.next_track_id = 0

        self.get_logger().info("ObjectTrackingNode initialized")
        self.get_logger().info(f"Listening on topic: {self.detection_topic}")

    def load_ros_params(self):
        """Get parameters from the ROS parameter server."""
        self.declare_parameters(
            namespace="",
            parameters=[
                ("debug", True),
                ("detection_topic", "/object_detection/object"),
                ("tracking_output_topic", "/object_tracking/tracked_objects"),
            ],
        )

        self.debug = self.get_parameter("debug").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.tracking_output_topic = self.get_parameter("tracking_output_topic").value

    def init_subscriber(self):
        """Initialize the subscriber for object detections."""
        self.detection_subscriber = self.create_subscription(
            Float32MultiArray, self.detection_topic, self.detection_callback, 10
        )

        # Initialize the publisher for tracked objects
        self.tracking_publisher = self.create_publisher(
            Float32MultiArray, self.tracking_output_topic, 10
        )

    def detection_callback(self, msg: Float32MultiArray):
        """
        Callback executed when new detections are received.

        Args:
            msg: Float32MultiArray containing detection data
        """
        if self.debug:
            self.get_logger().info("=" * 60)
            self.get_logger().info("Received detection message!")
            self.get_logger().info(f"Data length: {len(msg.data)}")

        # Parse the detection data
        detections = self.parse_detections(msg)

        if self.debug and detections:
            self.get_logger().info(f"Parsed {len(detections)} detection(s):")
            for i, det in enumerate(detections):
                self.get_logger().info(f"  Detection {i}: {det}")

        # Create mock tracked objects from detections
        tracked_objects = self.create_mock_tracks(detections)

        # Publish tracked objects
        if tracked_objects:
            tracking_msg = self.create_tracking_message(tracked_objects)
            self.tracking_publisher.publish(tracking_msg)

            if self.debug:
                self.get_logger().info(
                    f"Published {len(tracked_objects)} tracked object(s)"
                )
                for track in tracked_objects:
                    self.get_logger().info(
                        f"  Track ID {track['track_id']}: "
                        f"pos=({track['center']['x']:.1f}, {track['center']['y']:.1f}), "
                        f"confidence={track['confidence']:.3f}"
                    )

    def parse_detections(self, msg: Float32MultiArray):
        """
        Parse the Float32MultiArray into individual detections.

        Format from object_detection_node.py:
        [class_id, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, score]

        Where:
        - class_id: Object class identifier (int)
        - bottom_left_x/y: Bottom-left corner in world coordinates (mm or cm)
        - bottom_right_x/y: Bottom-right corner in world coordinates (mm or cm)
        - score: Detection confidence (0.0 - 1.0)

        Args:
            msg: Float32MultiArray message

        Returns:
            List of detection dictionaries
        """
        data = msg.data

        if len(data) == 0:
            if self.debug:
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
            # Calculate center and dimensions from corner points
            bottom_left_x = data[i + 1]
            bottom_left_y = data[i + 2]
            bottom_right_x = data[i + 3]
            bottom_right_y = data[i + 4]

            # Center point (average of two bottom corners)
            center_x = (bottom_left_x + bottom_right_x) / 2.0
            center_y = (bottom_left_y + bottom_right_y) / 2.0

            # Width from distance between left and right corners
            width = abs(bottom_right_x - bottom_left_x)

            detection = {
                "class_id": int(data[i]),
                "score": data[i + 5],
                "bottom_left": {"x": bottom_left_x, "y": bottom_left_y},
                "bottom_right": {"x": bottom_right_x, "y": bottom_right_y},
                "center": {"x": center_x, "y": center_y},
                "width": width,
            }
            detections.append(detection)

        return detections

    def create_mock_tracks(self, detections):
        """
        Create mock tracked objects from detections.

        For now, this is a simple mock implementation:
        - Each detection gets a new track ID
        - In a real implementation, you would use a tracking algorithm
          to associate detections across frames

        Args:
            detections: List of detection dictionaries

        Returns:
            List of tracked object dictionaries
        """
        tracked_objects = []

        for detection in detections:
            # Create a tracked object
            tracked_obj = {
                "track_id": self.next_track_id,
                "class_id": detection["class_id"],
                "center": detection["center"],
                "width": detection["width"],
                "confidence": detection["score"],
                "bottom_left": detection["bottom_left"],
                "bottom_right": detection["bottom_right"],
            }

            tracked_objects.append(tracked_obj)

            # Increment track ID for next object
            # TODO: In a real implementation, track IDs should persist across frames
            self.next_track_id += 1

        return tracked_objects

    def create_tracking_message(self, tracked_objects):
        """
        Create a Float32MultiArray message from tracked objects.

        Message format per tracked object:
        [track_id, center_x, center_y, confidence, class_id, width]

        Args:
            tracked_objects: List of tracked object dictionaries

        Returns:
            Float32MultiArray message
        """
        msg = Float32MultiArray()
        data = []

        for obj in tracked_objects:
            data.extend(
                [
                    float(obj["track_id"]),
                    float(obj["center"]["x"]),
                    float(obj["center"]["y"]),
                    float(obj["confidence"]),
                    float(obj["class_id"]),
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
