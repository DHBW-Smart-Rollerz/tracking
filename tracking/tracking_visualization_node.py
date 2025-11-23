"""
Tracking Visualization Node - PIL-based (like Object Detection).

Visualizes both tracked objects (cars, pedestrians) AND tracked signs on the same image.
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


class TrackingVisualizationNode(SmartyNode):
    """ROS2 Node for visualizing tracked objects AND signs using PIL."""

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
                "object_tracking_subscriber": "/object_tracking/tracked_objects",
                "sign_tracking_subscriber": "/sign_tracking/tracked_signs",
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
        self.latest_object_tracks = []  # Tracked objects
        self.latest_sign_tracks = []  # Tracked signs

        # Track timeout handling
        self.last_object_track_time = None
        self.last_sign_track_time = None
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
            f"Listening to object tracks: {self.subscribed_topics['object_tracking_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Listening to sign tracks: {self.subscribed_topics['sign_tracking_subscriber'][0]}"
        )
        self.get_logger().info(
            f"Publishing to: {self.published_topics['debug_image_publisher'][0]}"
        )

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

    def object_tracking_callback(self, msg: Float32MultiArray):
        """
        Callback for tracked objects.

        Args:
            msg: Float32MultiArray with tracking data
        """
        # Always update the list (including clearing it if empty)
        self.latest_object_tracks = self.parse_tracks(msg)
        self.last_object_track_time = self.get_clock().now()

    def sign_tracking_callback(self, msg: Float32MultiArray):
        """
        Callback for tracked signs.

        Args:
            msg: Float32MultiArray with tracking data
        """
        # Always update the list (including clearing it if empty)
        self.latest_sign_tracks = self.parse_tracks(msg)
        self.last_sign_track_time = self.get_clock().now()

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
        
    def world_to_pixel(self, x_world, y_world):
        """
        Convert a single world point (x, y) to pixel coordinates (u, v).
        """
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

    def draw_track(self, draw, track, detections, is_sign, font_large, font_normal, coord_offset=0):
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
            draw.rectangle([(xmin+3, ymin+3), (xmax-3, ymax-3)], outline=color, width=1)
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

        # --- NEW: Velocity Vector Visualization ---
        if self.show_velocity and not is_sign:
            # 1. Get start point (Current Track Position)
            # We use the track state, NOT the detection, because we want to see the Filter's belief
            cx, cy = self.world_to_pixel(track["x"], track["y"])
            
            # 2. Get end point (Projected Position 1 second in the future)
            # Scaling factor: How long the arrow should look (1.0 = 1 second of movement)
            arrow_scale = 1.0 
            fx, fy = self.world_to_pixel(
                track["x"] + track["vx"] * arrow_scale, 
                track["y"] + track["vy"] * arrow_scale
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
            r = 4 # radius
            draw.ellipse([(fx-r, fy-r), (fx+r, fy+r)], fill="cyan")

    def visualize_and_publish(self):
        """Draw tracked objects and signs on image using PIL and publish."""
        if self.latest_image is None:
            return

        # Check for track timeouts and clear if needed
        current_time = self.get_clock().now()
        
        if self.last_object_track_time is not None:
            time_diff = (current_time - self.last_object_track_time).nanoseconds / 1e9
            if time_diff > self.track_timeout_sec:
                self.latest_object_tracks = []
                
        if self.last_sign_track_time is not None:
            time_diff = (current_time - self.last_sign_track_time).nanoseconds / 1e9
            if time_diff > self.track_timeout_sec:
                self.latest_sign_tracks = []

        try:
            # Get original image dimensions
            orig_height, orig_width = self.latest_image.shape[:2]
            
            # Create expanded canvas (add padding for boxes extending beyond bounds)
            padding = 200  # pixels of padding on all sides
            expanded_height = orig_height + 2 * padding
            expanded_width = orig_width + 2 * padding
            
            # Create expanded image with black background
            expanded_image = np.zeros((expanded_height, expanded_width, 3), dtype=np.uint8)
            
            # Paste original image in center
            expanded_image[padding:padding+orig_height, padding:padding+orig_width] = self.latest_image
            
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
            stats_text = f"Objects: {len(self.latest_object_tracks)} | Signs: {len(self.latest_sign_tracks)} | Frame: {self.frame_count}"
            draw.rectangle([(5+padding, 5+padding), (350+padding, 30+padding)], fill="black", outline="white")
            draw.text((10+padding, 10+padding), stats_text, fill="white", font=font_normal)

            # Legend
            legend_y = 40 + padding
            draw.text(
                (10+padding, legend_y),
                "O# = Object Track TEST | S# = Sign Track",
                fill="white",
                font=font_normal,
            )

            # Convert back to numpy
            result_image = np.array(pil_image)
            
            # Crop back to original size (remove padding)
            result_image = result_image[padding:padding+orig_height, padding:padding+orig_width]

            # Convert RGB to BGR for ROS
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

            output_msg = self.cv_bridge.cv2_to_imgmsg(result_image_bgr, encoding="bgr8")
            self.debug_image_publisher.publish(output_msg)  # type: ignore
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"Published frame {self.frame_count} with "
                    f"{len(self.latest_object_tracks)} object tracks, "
                    f"{len(self.latest_sign_tracks)} sign tracks"
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