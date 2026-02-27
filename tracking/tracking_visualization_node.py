"""
Tracking Visualization Node - PIL-based (like Object Detection).

Visualizes tracked objects (cars, pedestrians), tracked signs, AND tracked
crossing lines on the same image.
Uses separate colors/styles to distinguish between object types.
"""

import cv2
import cv_bridge
import numpy as np
import rclpy
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import Image
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from std_msgs.msg import Float32MultiArray

# Import coordinate transformation
try:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
        Unit,
    )

    COORD_TRANSFORM_AVAILABLE = True
except ImportError:
    COORD_TRANSFORM_AVAILABLE = False


# LaneType class IDs from crossing detection
LANE_TYPE_EGO_SOLID = 19
LANE_TYPE_EGO_DOTTED = 20
LANE_TYPE_OPP_SOLID = 21
LANE_TYPE_OPP_DOTTED = 22


class TrackingVisualizationNode(SmartyNode):
    """ROS2 Node for visualizing tracked objects, signs, AND crossings using PIL."""

    def __init__(self):
        """Initialize the TrackingVisualizationNode."""
        super().__init__(
            "tracking_visualization_node",
            "tracking",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/image/undistorted",
                "object_detection_subscriber": "/object_detection/object",
                "sign_detection__subscriber": "/object_detection/sign",
                "crossing_detection_subscriber": "/crossing_detection/result",
                "object_tracking_subscriber": "/object_tracking/tracked_objects",
                "sign_tracking_subscriber": "/sign_tracking/tracked_signs",
                "crossing_tracking_subscriber": "/crossing_tracking/tracked_crossings",
                # Publisher topics
                "debug_image_publisher": "/tracking/debug/image",
                # Parameters
                "state": NodeState.ACTIVE.value,
                "show_velocity": True,
                "text_size": 12,
                "debug": False,
            },
            subscribed_topics={
                "image_subscriber": (
                    Image,
                    self.image_callback,
                    1,
                ),
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
                "object_tracking_subscriber": (
                    Float32MultiArray,
                    self.object_tracking_callback,
                    1,
                ),
                "sign_tracking_subscriber": (
                    Float32MultiArray,
                    self.sign_tracking_callback,
                    1,
                ),
                "crossing_tracking_subscriber": (
                    Float32MultiArray,
                    self.crossing_tracking_callback,
                    1,
                ),
            },
            published_topics={
                "debug_image_publisher": (Image, 1),
            },
        )
        # Initialize CV bridge
        self.cv_bridge = cv_bridge.CvBridge()

        # Storage for latest messages
        self.latest_image = None
        self.latest_object_detections = []  # For objects (cars, pedestrians)
        self.latest_sign_detections = []  # For signs
        self.latest_crossing_detections = []  # For crossing lines
        self.latest_object_tracks = []  # Tracked objects
        self.latest_sign_tracks = []  # Tracked signs
        self.latest_crossing_tracks = []  # Tracked crossings

        # Track timeout handling
        self.last_object_track_time = None
        self.last_sign_track_time = None
        self.last_crossing_track_time = None
        self.track_timeout_sec = 0.5  # Clear tracks if no message for 0.5 seconds

        # Statistics
        self.frame_count = 0

        # Class ID to name mapping
        self.class_names = {
            1: "Stop",
            2: "Car",
            7: "30",
            8: "No30",
            9: "Cross",
            10: "Ped",
            14: "Park",
            15: "Left",
            16: "Right",
            17: "Priority",
            18: "Yield",
            # Crossing lane types
            LANE_TYPE_EGO_SOLID: "Ego Solid",
            LANE_TYPE_EGO_DOTTED: "Ego Dotted",
            LANE_TYPE_OPP_SOLID: "Opp Solid",
            LANE_TYPE_OPP_DOTTED: "Opp Dotted",
        }

        # Crossing line colors: (line_color, label_color) per LaneType
        self.crossing_colors = {
            LANE_TYPE_EGO_SOLID: "#00DD00",  # bright green
            LANE_TYPE_EGO_DOTTED: "#00DD00",  # bright green (dashed)
            LANE_TYPE_OPP_SOLID: "#4488FF",  # bright blue
            LANE_TYPE_OPP_DOTTED: "#4488FF",  # bright blue (dashed)
        }

        self.get_logger().info("TrackingVisualizationNode initialized (PIL-based)")
        self.get_logger().info(
            f"Listening to image: {self.subscribed_topics['image_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to object detections: {self.subscribed_topics['object_detection_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to sign detections: {self.subscribed_topics['sign_detection__subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to crossing detections: {self.subscribed_topics['crossing_detection_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to object tracks: {self.subscribed_topics['object_tracking_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to sign tracks: {self.subscribed_topics['sign_tracking_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to crossing tracks: {self.subscribed_topics['crossing_tracking_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Publishing to: {self.published_topics['debug_image_publisher'][0]}"
        )

        # Initializing the timer for fps calculation
        self.last_frame_time = self.get_clock().now()
        self.fps = 0.0

        self.last_tracking_time = None
        self.tracking_fps = 0.0

        # Initialize coordinate transformation
        if COORD_TRANSFORM_AVAILABLE:
            try:
                self.coord_transform = CoordinateTransform(debug=False)
                self.get_logger().info("CoordinateTransform initialized successfully")
            except Exception as e:
                self.get_logger().warn(f"Failed to initialize CoordinateTransform: {e}")
                self.coord_transform = None
        else:
            self.get_logger().warn(
                "CoordinateTransform not available - using fallback transformation"
            )
            self.coord_transform = None

    @property
    def show_velocity(self) -> bool:
        """Return whether to show velocity on visualization."""
        return self.get_parameter("show_velocity").value  # type: ignore

    @property
    def text_size(self) -> int:
        """Return text size for visualization."""
        return self.get_parameter("text_size").value  # type: ignore

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def image_callback(self, msg: Image):
        """
        Callback for camera image.

        Args:
            msg: Image message
        """
        try:
            # Convert to numpy array (grayscale to RGB for PIL)
            if msg.encoding in ["mono8", "8UC1"]:
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width)
                )
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            self.latest_image = image

            # Always visualize on new image (even if no tracks)
            # This ensures bounding boxes disappear when tracks are deleted
            if self.latest_image is not None:
                self.visualize_and_publish()

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def object_detection_callback(self, msg: Float32MultiArray):
        """
        Callback for object detections (cars, pedestrians).

        Args:
            msg: Float32MultiArray with detection data
        """
        self.latest_object_detections = self.parse_detections(msg)

    def sign_detection_callback(self, msg: Float32MultiArray):
        """
        Callback for sign detections.

        Args:
            msg: Float32MultiArray with detection data
        """
        self.latest_sign_detections = self.parse_detections(msg)

    def crossing_detection_callback(self, msg: Float32MultiArray):
        """
        Callback for crossing detections (ego/opp lane lines).

        Args:
            msg: Float32MultiArray with detection data
        """
        self.latest_crossing_detections = self.parse_detections(msg)

    def object_tracking_callback(self, msg: Float32MultiArray):
        """
        Callback for tracked objects.

        Args:
            msg: Float32MultiArray with tracking data
        """
        current_time = self.get_clock().now()

        # FPS calculation for tracking messages
        if self.last_tracking_time is not None:
            duration = (current_time - self.last_tracking_time).nanoseconds / 1e9
            if duration > 0:
                actual_fps = 1.0 / duration
                self.tracking_fps = (self.tracking_fps * 0.9) + (actual_fps * 0.1)
        self.last_tracking_time = current_time

        # Always update the list (including clearing it if empty)
        parsed_tracks = self.parse_tracks(msg)
        self.latest_object_tracks = parsed_tracks
        self.last_object_track_time = current_time

    def sign_tracking_callback(self, msg: Float32MultiArray):
        """
        Callback for tracked signs.

        Args:
            msg: Float32MultiArray with tracking data
        """
        # Always update the list (including clearing it if empty)
        self.latest_sign_tracks = self.parse_tracks(msg)
        self.last_sign_track_time = self.get_clock().now()

    def crossing_tracking_callback(self, msg: Float32MultiArray):
        """
        Callback for tracked crossings.

        Args:
            msg: Float32MultiArray with tracking data
        """
        self.latest_crossing_tracks = self.parse_tracks(msg)
        self.last_crossing_track_time = self.get_clock().now()

    # ------------------------------------------------------------------ #
    #  Message parsing                                                    #
    # ------------------------------------------------------------------ #

    def parse_detections(self, msg: Float32MultiArray):
        """
        Parse detection data.

        Format: [class_id, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, score]

        Args:
            msg: Float32MultiArray message

        Returns:
            List of detection dictionaries with world coordinates
        """
        data = msg.data
        if len(data) == 0:
            return []

        values_per_detection = 6
        if len(data) % values_per_detection != 0:
            return []

        detections = []
        for i in range(0, len(data), values_per_detection):
            detection = {
                "class_id": int(data[i]),
                "bottom_left": {"x": data[i + 1], "y": data[i + 2]},
                "bottom_right": {"x": data[i + 3], "y": data[i + 4]},
                "score": data[i + 5],
                # Calculate center
                "center": {
                    "x": (data[i + 1] + data[i + 3]) / 2.0,
                    "y": (data[i + 2] + data[i + 4]) / 2.0,
                },
            }
            detections.append(detection)

        return detections

    def parse_tracks(self, msg: Float32MultiArray):
        """
        Parse tracked objects.

        Format: [track_id, class_id, x, y, vx, vy, confidence, width]

        Args:
            msg: Float32MultiArray message

        Returns:
            List of tracked object dictionaries
        """
        data = msg.data
        if len(data) == 0:
            return []

        values_per_track = 8
        if len(data) % values_per_track != 0:
            return []

        tracks = []
        for i in range(0, len(data), values_per_track):
            track = {
                "track_id": int(data[i]),
                "class_id": int(data[i + 1]),
                "x": data[i + 2],
                "y": data[i + 3],
                "vx": data[i + 4],
                "vy": data[i + 5],
                "confidence": data[i + 6],
                "width": data[i + 7],
            }
            tracks.append(track)

        return tracks

    # ------------------------------------------------------------------ #
    #  Track-to-detection matching                                        #
    # ------------------------------------------------------------------ #

    def match_track_to_detection(self, track, detections):
        """
        Find the detection that best matches a track.

        Args:
            track: Track dictionary
            detections: List of detection dictionaries

        Returns:
            Matching detection or None
        """
        if not detections:
            return None

        # Find detection with same class and closest position
        best_match = None
        min_distance = float("inf")

        for detection in detections:
            # Check if class matches
            if detection["class_id"] != track["class_id"]:
                continue

            # Calculate distance

            dx = detection["center"]["x"] - track["x"]
            dy = detection["center"]["y"] - track["y"]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < min_distance and distance < 500:  # 500mm threshold
                min_distance = distance
                best_match = detection

        return best_match

    # ------------------------------------------------------------------ #
    #  Coordinate transformations                                         #
    # ------------------------------------------------------------------ #

    def world_coords_to_bbox(self, track, image_shape):
        """
        Convert world coordinates to bounding box using CoordinateTransform.

        Args:
            track: Track dictionary with world coordinates
            image_shape: (height, width, channels)

        Returns:
            Tuple (xmin, ymin, xmax, ymax) in pixels
        """
        if self.coord_transform is not None:
            try:
                # Create center point in world coordinates
                center_point = np.array([[track["x"], track["y"], 0]])

                # Transform to pixel coordinates
                center_pixel = self.coord_transform.world_to_camera(
                    center_point, input_unit=Unit.MILLIMETERS
                )[0]

                # Estimate bounding box size from width
                # Scale width from mm to pixels (approximate)
                width_pixels = int(track["width"] * 0.1)  # Rough scaling
                height_pixels = int(width_pixels * 1.3)

                # Create bounding box around center
                center_x, center_y = int(center_pixel[0]), int(center_pixel[1])
                xmin = center_x - width_pixels // 2
                xmax = center_x + width_pixels // 2
                ymin = center_y - height_pixels // 2
                ymax = center_y + height_pixels // 2

                # Don't clamp - allow boxes to extend beyond image bounds
                return xmin, ymin, xmax, ymax

            except Exception as e:
                self.get_logger().debug(
                    f"CoordinateTransform failed: {e}, using fallback"
                )

        # Fallback transformation
        return self._fallback_world_to_bbox(track, image_shape)

    def _fallback_world_to_bbox(self, track, image_shape):
        """
        Fallback transformation when CoordinateTransform is not available.

        Args:
            track: Track dictionary
            image_shape: (height, width, channels)

        Returns:
            Tuple (xmin, ymin, xmax, ymax) in pixels
        """
        height, width = image_shape[:2]

        # Simple estimation
        scale = 0.08
        center_x = width // 2
        center_y = height - 80

        # Transform center point
        px = int(center_x + track["y"] * scale)
        py = int(center_y - track["x"] * scale)

        # Estimate box size
        box_width = max(20, int(track["width"] * scale))
        box_height = int(box_width * 1.3)

        xmin = px - box_width // 2
        xmax = px + box_width // 2
        ymin = py - box_height // 2
        ymax = py + box_height // 2

        # Don't clamp - allow boxes to extend beyond image bounds
        return xmin, ymin, xmax, ymax

    def detection_to_bbox(self, detection, image_shape):
        """
        Convert detection world coordinates to pixel bounding box.

        Args:
            detection: Detection dictionary with world coordinates
            image_shape: (height, width, channels)

        Returns:
            Tuple (xmin, ymin, xmax, ymax) in pixels
        """
        if self.coord_transform is not None:
            try:
                # Create corner points in world coordinates
                bottom_left = np.array(
                    [[detection["bottom_left"]["x"], detection["bottom_left"]["y"], 0]]
                )

                bottom_right = np.array(
                    [
                        [
                            detection["bottom_right"]["x"],
                            detection["bottom_right"]["y"],
                            0,
                        ]
                    ]
                )

                # Transform to pixel coordinates
                bl_pixel = self.coord_transform.world_to_camera(
                    bottom_left, input_unit=Unit.MILLIMETERS
                )[0]

                br_pixel = self.coord_transform.world_to_camera(
                    bottom_right, input_unit=Unit.MILLIMETERS
                )[0]

                # Create bounding box from corners
                xmin = int(min(bl_pixel[0], br_pixel[0]))
                xmax = int(max(bl_pixel[0], br_pixel[0]))

                # Estimate height (assume rectangular box)
                width = xmax - xmin
                height = int(width * 1.3)  # Approximate height from width

                # Bottom corners define ymax
                ymax = int(max(bl_pixel[1], br_pixel[1]))
                ymin = ymax - height

                # Don't clamp - allow boxes to extend beyond image bounds
                return xmin, ymin, xmax, ymax

            except Exception as e:
                self.get_logger().debug(
                    f"CoordinateTransform failed: {e}, using fallback"
                )

        # Fallback to simple transformation
        return self._fallback_detection_to_bbox(detection, image_shape)

    def _fallback_detection_to_bbox(self, detection, image_shape):
        """
        Fallback transformation when CoordinateTransform is not available.

        Args:
            detection: Detection dictionary with world coordinates
            image_shape: (height, width, channels)

        Returns:
            Tuple (xmin, ymin, xmax, ymax) in pixels
        """
        height, width = image_shape[:2]

        # Simple estimation
        scale = 0.08
        center_x = width // 2
        center_y = height - 80

        # Bottom-left corner
        bl_x = int(center_x + detection["bottom_left"]["y"] * scale)
        bl_y = int(center_y - detection["bottom_left"]["x"] * scale)

        # Bottom-right corner
        br_x = int(center_x + detection["bottom_right"]["y"] * scale)
        br_y = int(center_y - detection["bottom_right"]["x"] * scale)

        # Create bounding box from corners
        xmin = min(bl_x, br_x)
        xmax = max(bl_x, br_x)
        ymin = min(bl_y, br_y) - int((xmax - xmin) * 1.2)
        ymax = max(bl_y, br_y)

        # Don't clamp - allow boxes to extend beyond image bounds
        return xmin, ymin, xmax, ymax

    def detection_to_line_pixels(self, detection, bev=False):
        """
        Convert a crossing detection's two endpoints to pixel coordinates.

        For crossing detections, bottom_left and bottom_right represent
        the two endpoints of the detected lane line (not a bounding box).

        Args:
            detection: Detection dictionary with endpoint coordinates
            bev: If True, treat coordinates as BEV pixels and use bird_to_camera.
                 If False, treat as world coordinates and use world_to_pixel.

        Returns:
            Tuple ((x1, y1), (x2, y2)) in camera pixel coordinates, or None on failure
        """
        x1 = detection["bottom_left"]["x"]
        y1 = detection["bottom_left"]["y"]
        x2 = detection["bottom_right"]["x"]
        y2 = detection["bottom_right"]["y"]

        if bev:
            p1 = self.bev_to_pixel(x1, y1)
            p2 = self.bev_to_pixel(x2, y2)
            if p1 is None or p2 is None:
                return None
        else:
            p1 = self.world_to_pixel(x1, y1)
            p2 = self.world_to_pixel(x2, y2)

        return p1, p2

    # ------------------------------------------------------------------ #
    #  Color helpers                                                      #
    # ------------------------------------------------------------------ #

    def get_color_from_confidence(self, confidence):
        """
        Get color based on confidence level.

        Args:
            confidence: Float between 0 and 1

        Returns:
            Color string for PIL
        """
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.6:
            return "yellow"
        elif confidence >= 0.4:
            return "orange"
        else:
            return "red"

    def get_crossing_color(self, class_id):
        """
        Get color for a crossing line based on its LaneType class ID.

        Args:
            class_id: LaneType enum value (19-22)

        Returns:
            Color string for PIL
        """
        return self.crossing_colors.get(class_id, "#FFFF00")  # yellow fallback

    def is_crossing_dotted(self, class_id):
        """
        Check if a crossing lane type is dotted.

        Args:
            class_id: LaneType enum value

        Returns:
            True if the lane type is dotted
        """
        return class_id in (LANE_TYPE_EGO_DOTTED, LANE_TYPE_OPP_DOTTED)

    # ------------------------------------------------------------------ #
    #  Pixel coordinate helpers                                           #
    # ------------------------------------------------------------------ #

    def world_to_pixel(self, x_world, y_world):
        """Convert a single world point (x, y) to pixel coordinates (u, v)."""
        # 1. Try CoordinateTransform
        if self.coord_transform is not None:
            try:
                point = np.array([[x_world, y_world, 0]])
                pixel = self.coord_transform.world_to_camera(
                    point, input_unit=Unit.MILLIMETERS
                )[0]
                return int(pixel[0]), int(pixel[1])
            except Exception:
                pass  # Fallback if transform fails

        # 2. Fallback (Same logic as _fallback_world_to_bbox)
        height, width = self.latest_image.shape[:2]
        scale = 0.08
        center_x = width // 2
        center_y = height - 80

        px = int(center_x + y_world * scale)
        py = int(center_y - x_world * scale)
        return px, py

    def bev_to_pixel(self, x_bev, y_bev):
        """Convert a single BEV (bird's-eye view) pixel coordinate to camera pixel coordinates.

        The crossing detection publishes line endpoints in BEV pixel space.
        This method uses CoordinateTransform.bird_to_camera() to map them
        back onto the undistorted camera image.

        Args:
            x_bev: X coordinate in BEV image (pixels)
            y_bev: Y coordinate in BEV image (pixels)

        Returns:
            Tuple (u, v) in camera pixel coordinates, or None on failure
        """
        if self.coord_transform is not None:
            try:
                bev_point = np.array([[x_bev, y_bev]])
                cam_point = self.coord_transform.bird_to_camera(bev_point)[0]
                return int(cam_point[0]), int(cam_point[1])
            except Exception as e:
                self.get_logger().debug(f"bird_to_camera failed: {e}")

        # Fallback: no transformation available — return None so caller can handle it
        self.get_logger().warn(
            "bev_to_pixel: CoordinateTransform not available, cannot map crossing"
        )
        return None

    # ------------------------------------------------------------------ #
    #  Drawing: objects & signs (bounding boxes)                          #
    # ------------------------------------------------------------------ #

    def draw_track(
        self, draw, track, detections, is_sign, font_large, font_normal, coord_offset=0
    ):
        """
        Draw a single tracked object on the image.

        Args:
            draw: PIL ImageDraw object
            track: Track dictionary
            detections: List of detections to match against
            is_sign: Boolean, True if this is a sign track
            font_large: Large font for track ID
            font_normal: Normal font for other text
            coord_offset: Pixel offset to apply to all coordinates (for expanded canvas)
        """
        # Try to match with detection for better pixel coordinates
        matched_detection = self.match_track_to_detection(track, detections)

        if matched_detection is not None:
            # Use detection's pixel coordinates
            xmin, ymin, xmax, ymax = self.detection_to_bbox(
                matched_detection, self.latest_image.shape  # type: ignore
            )
        else:
            # Estimate from track world coordinates
            xmin, ymin, xmax, ymax = self.world_coords_to_bbox(
                track, self.latest_image.shape  # type: ignore
            )

        # Apply coordinate offset for expanded canvas
        xmin += coord_offset
        xmax += coord_offset
        ymin += coord_offset
        ymax += coord_offset

        # Get color based on confidence
        color = self.get_color_from_confidence(track["confidence"])

        # Draw bounding box - simple like Object Detection!
        # Signs: double outline for distinction
        # Objects: single outline
        if is_sign:
            # Draw outer box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
            # Draw inner box for double-outline effect
            draw.rectangle(
                [(xmin + 3, ymin + 3), (xmax - 3, ymax - 3)], outline=color, width=1
            )
        else:
            # Objects: solid single box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)

        # Draw Track ID (LARGE and prominent)
        prefix = "S" if is_sign else "O"
        track_text = f"{prefix}#{track['track_id']}"
        draw.text((xmin + 5, ymin + 5), track_text, fill=color, font=font_large)

        # Draw class name
        class_name = self.class_names.get(track["class_id"], f"ID{track['class_id']}")
        draw.text((xmin + 5, ymin + 25), class_name, fill=color, font=font_normal)

        # Draw confidence
        conf_text = f"{track['confidence']:.2f}"
        draw.text((xmin + 5, ymin + 40), conf_text, fill=color, font=font_normal)

        # Draw velocity (if enabled and not a sign)
        if self.show_velocity and not is_sign:
            speed = np.sqrt(track["vx"] ** 2 + track["vy"] ** 2)
            if speed > 50:  # Only show if moving
                vel_text = f"{int(speed)} mm/s"
                draw.text((xmin + 5, ymin + 55), vel_text, fill=color, font=font_normal)

        # --- Velocity Vector Visualization ---
        if self.show_velocity and not is_sign:
            # 1. Get start point (Current Track Position)
            # We use the track state, NOT the detection, because we want to see the Filter's belief
            cx, cy = self.world_to_pixel(track["x"], track["y"])

            # 2. Get end point (Projected Position 1 second in the future)
            # Scaling factor: How long the arrow should look (1.0 = 1 second of movement)
            arrow_scale = 1.0
            fx, fy = self.world_to_pixel(
                track["x"] + track["vx"] * arrow_scale,
                track["y"] + track["vy"] * arrow_scale,
            )

            # 3. Apply the padding offset (since we are drawing on the expanded canvas)
            cx += coord_offset
            cy += coord_offset
            fx += coord_offset
            fy += coord_offset

            # 4. Draw the Line (Shaft)
            # We use a thick line for visibility
            draw.line([(cx, cy), (fx, fy)], fill="cyan", width=3)

            # 5. Draw a simple Circle at the tip (Head)
            # (easier than calculating a rotated triangle)
            r = 4  # radius
            draw.ellipse([(fx - r, fy - r), (fx + r, fy + r)], fill="cyan")

    # ------------------------------------------------------------------ #
    #  Drawing: crossing lines                                            #
    # ------------------------------------------------------------------ #

    def draw_dashed_line(
        self, draw, p1, p2, color, width=3, dash_length=12, gap_length=8
    ):
        """
        Draw a dashed line between two points using PIL.

        Args:
            draw: PIL ImageDraw object
            p1: Start point (x, y)
            p2: End point (x, y)
            color: Line color
            width: Line width in pixels
            dash_length: Length of each dash in pixels
            gap_length: Length of each gap in pixels
        """
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return

        # Unit direction vector
        ux = dx / length
        uy = dy / length

        segment_length = dash_length + gap_length
        pos = 0.0

        while pos < length:
            # Start of this dash
            sx = x1 + ux * pos
            sy = y1 + uy * pos

            # End of this dash (clamp to total length)
            end_pos = min(pos + dash_length, length)
            ex = x1 + ux * end_pos
            ey = y1 + uy * end_pos

            draw.line(
                [(int(sx), int(sy)), (int(ex), int(ey))],
                fill=color,
                width=width,
            )

            pos += segment_length

    def draw_crossing_track(
        self, draw, track, detections, font_large, font_normal, coord_offset=0
    ):
        """
        Draw a single tracked crossing line on the image.

        Instead of a bounding box, this draws the actual line between the two
        detection endpoints, color-coded by LaneType:
        - EGO lines: green (solid or dashed)
        - OPP lines: blue (solid or dashed)

        If no matching detection is found, falls back to drawing a marker
        at the track's center position.

        Args:
            draw: PIL ImageDraw object
            track: Track dictionary
            detections: List of crossing detection dictionaries
            font_large: Large font for track ID
            font_normal: Normal font for other text
            coord_offset: Pixel offset for expanded canvas
        """
        class_id = track["class_id"]
        color = self.get_crossing_color(class_id)
        is_dotted = self.is_crossing_dotted(class_id)
        class_name = self.class_names.get(class_id, f"Lane{class_id}")

        # Try to match track to a raw detection to get the line endpoints
        matched_detection = self.match_track_to_detection(track, detections)

        if matched_detection is not None:
            # We have endpoints — draw the actual line
            # Crossing detections are in BEV pixel coordinates
            result = self.detection_to_line_pixels(matched_detection, bev=True)

            if result is None:
                # bird_to_camera failed — skip this track
                self.get_logger().debug(
                    f"Crossing C#{track['track_id']}: BEV→camera transform failed"
                )
                return

            p1, p2 = result

            # Apply padding offset
            p1 = (p1[0] + coord_offset, p1[1] + coord_offset)
            p2 = (p2[0] + coord_offset, p2[1] + coord_offset)

            # Draw solid or dashed line
            if is_dotted:
                self.draw_dashed_line(draw, p1, p2, color=color, width=4)
            else:
                draw.line([p1, p2], fill=color, width=4)

            # Small circles at endpoints for clarity
            r = 4
            draw.ellipse([(p1[0] - r, p1[1] - r), (p1[0] + r, p1[1] + r)], fill=color)
            draw.ellipse([(p2[0] - r, p2[1] - r), (p2[0] + r, p2[1] + r)], fill=color)

            # Label at midpoint
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

        else:
            # No detection match — fallback: draw a diamond marker at track center
            # Track coordinates are also in BEV pixel space
            pixel = self.bev_to_pixel(track["x"], track["y"])
            if pixel is None:
                self.get_logger().debug(
                    f"Crossing C#{track['track_id']}: BEV→camera fallback failed"
                )
                return

            cx, cy = pixel
            cx += coord_offset
            cy += coord_offset

            # Diamond shape around center
            size = 10
            diamond = [
                (cx, cy - size),
                (cx + size, cy),
                (cx, cy + size),
                (cx - size, cy),
            ]
            draw.polygon(diamond, outline=color, fill=None)

            # If dotted type, add inner dot to distinguish
            if is_dotted:
                r = 3
                draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)

            mid_x = cx
            mid_y = cy

        # Draw label: track ID + class name + confidence
        label_y = mid_y - 25  # above the line midpoint
        track_text = f"C#{track['track_id']}"
        draw.text((mid_x + 5, label_y), track_text, fill=color, font=font_large)
        draw.text(
            (mid_x + 5, label_y + 16),
            f"{class_name} ({track['confidence']:.2f})",
            fill=color,
            font=font_normal,
        )

    # ------------------------------------------------------------------ #
    #  Main visualization loop                                            #
    # ------------------------------------------------------------------ #

    def visualize_and_publish(self):
        """Draw tracked objects, signs, and crossings on image using PIL and publish."""
        if self.latest_image is None:
            return

        # Check for track timeouts and clear if needed
        current_time = self.get_clock().now()

        # FPS calculation
        time_diff = (current_time - self.last_frame_time).nanoseconds / 1e9
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.fps = (self.fps * 0.9) + (current_fps * 0.1)
        self.last_frame_time = current_time

        if self.last_object_track_time is not None:
            time_diff = (current_time - self.last_object_track_time).nanoseconds / 1e9
            if time_diff > self.track_timeout_sec:
                self.latest_object_tracks = []

        if self.last_sign_track_time is not None:
            time_diff = (current_time - self.last_sign_track_time).nanoseconds / 1e9
            if time_diff > self.track_timeout_sec:
                self.latest_sign_tracks = []

        if self.last_crossing_track_time is not None:
            time_diff = (current_time - self.last_crossing_track_time).nanoseconds / 1e9
            if time_diff > self.track_timeout_sec:
                self.latest_crossing_tracks = []

        try:
            # Get original image dimensions
            orig_height, orig_width = self.latest_image.shape[:2]

            # Create expanded canvas (add padding for boxes extending beyond bounds)
            padding = 200  # pixels of padding on all sides
            expanded_height = orig_height + 2 * padding
            expanded_width = orig_width + 2 * padding

            # Create expanded image with black background
            expanded_image = np.zeros(
                (expanded_height, expanded_width, 3), dtype=np.uint8
            )

            # Paste original image in center
            expanded_image[
                padding : padding + orig_height, padding : padding + orig_width
            ] = self.latest_image

            # Convert expanded numpy array to PIL Image
            pil_image = PILImage.fromarray(expanded_image)
            draw = ImageDraw.Draw(pil_image)

            # Try to load a font (fallback to default if not available)
            try:
                font_large = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    self.text_size + 8,
                )
                font_normal = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.text_size
                )
            except:
                font_large = ImageFont.load_default()
                font_normal = ImageFont.load_default()

            # Draw crossing lines FIRST (so they appear behind bounding boxes)
            for track in self.latest_crossing_tracks:
                self.draw_crossing_track(
                    draw,
                    track,
                    self.latest_crossing_detections,
                    font_large=font_large,
                    font_normal=font_normal,
                    coord_offset=padding,
                )

            # Draw tracked objects (cars, pedestrians) - adjust coords by padding
            for track in self.latest_object_tracks:
                self.draw_track(
                    draw,
                    track,
                    self.latest_object_detections,
                    is_sign=False,
                    font_large=font_large,
                    font_normal=font_normal,
                    coord_offset=padding,  # New parameter
                )

            # Draw tracked signs - adjust coords by padding
            for track in self.latest_sign_tracks:
                self.draw_track(
                    draw,
                    track,
                    self.latest_sign_detections,
                    is_sign=True,
                    font_large=font_large,
                    font_normal=font_normal,
                    coord_offset=padding,  # New parameter
                )

            # Draw statistics (on expanded canvas, so add padding offset)
            stats_text = (
                f"Objects: {len(self.latest_object_tracks)} | "
                f"Signs: {len(self.latest_sign_tracks)} | "
                f"Crossings: {len(self.latest_crossing_tracks)} | "
                f"Frame: {self.frame_count} | "
                f"Vis-FPS: {self.fps:.1f} | "
                f"Track FPS: {self.tracking_fps:.1f}"
            )
            draw.rectangle(
                [(5 + padding, 5 + padding), (560 + padding, 30 + padding)],
                fill="black",
                outline="white",
            )
            draw.text(
                (10 + padding, 10 + padding), stats_text, fill="white", font=font_normal
            )

            # Legend
            legend_y = 40 + padding
            draw.text(
                (10 + padding, legend_y),
                "O# = Object | S# = Sign | C# = Crossing TEST TEST",
                fill="white",
                font=font_normal,
            )
            # Crossing color legend
            legend_y += 16
            draw.line(
                [(10 + padding, legend_y + 6), (30 + padding, legend_y + 6)],
                fill=self.crossing_colors[LANE_TYPE_EGO_SOLID],
                width=2,
            )
            draw.text(
                (35 + padding, legend_y),
                "Ego",
                fill=self.crossing_colors[LANE_TYPE_EGO_SOLID],
                font=font_normal,
            )
            draw.line(
                [(70 + padding, legend_y + 6), (90 + padding, legend_y + 6)],
                fill=self.crossing_colors[LANE_TYPE_OPP_SOLID],
                width=2,
            )
            draw.text(
                (95 + padding, legend_y),
                "Opp",
                fill=self.crossing_colors[LANE_TYPE_OPP_SOLID],
                font=font_normal,
            )
            draw.text(
                (130 + padding, legend_y),
                "(solid = ——  dotted = - - -)",
                fill="white",
                font=font_normal,
            )

            # Convert back to numpy
            result_image = np.array(pil_image)

            # Crop back to original size (remove padding)
            result_image = result_image[
                padding : padding + orig_height, padding : padding + orig_width
            ]

            # Convert RGB to BGR for ROS
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

            output_msg = self.cv_bridge.cv2_to_imgmsg(result_image_bgr, encoding="bgr8")
            self.debug_image_publisher.publish(output_msg)  # type: ignore
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"Published frame {self.frame_count} with "
                    f"{len(self.latest_object_tracks)} object tracks, "
                    f"{len(self.latest_sign_tracks)} sign tracks, "
                    f"{len(self.latest_crossing_tracks)} crossing tracks"
                )

        except Exception as e:
            self.get_logger().error(f"Failed to visualize: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())


def main(args=None):
    """
    Main function to start the TrackingVisualizationNode.

    Args:
        args: Launch arguments (default: None)
    """
    rclpy.init(args=args)
    node = TrackingVisualizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Tracking Visualization Node")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
