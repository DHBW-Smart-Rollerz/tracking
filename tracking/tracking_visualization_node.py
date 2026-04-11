"""
Tracking Visualization Node - PIL-based.

Visualizes tracked objects, tracked signs, AND tracked crossing lines
from the unified state_msgs/State message on the same image.
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

# Import der neuen Custom Messages
import state_msgs.msg

# Import coordinate transformation
try:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
        Unit,
    )

    COORD_TRANSFORM_AVAILABLE = True
except ImportError:
    COORD_TRANSFORM_AVAILABLE = False


# LaneType class IDs from crossing detection (must match LaneType IntEnum)
LANE_TYPE_EGO_SOLID = 20
LANE_TYPE_EGO_DOTTED = 21
LANE_TYPE_OPP_SOLID = 22
LANE_TYPE_OPP_DOTTED = 23
LANE_TYPE_RIGHT_SOLID = 24
LANE_TYPE_RIGHT_DOTTED = 25
LANE_TYPE_LEFT_SOLID = 26
LANE_TYPE_LEFT_DOTTED = 27


class TrackingVisualizationNode(SmartyNode):
    """ROS2 Node for visualizing tracked objects, signs, and crossings using PIL."""

    def __init__(self):
        """Initialize the TrackingVisualizationNode."""
        super().__init__(
            "tracking_visualization_node",
            "tracking",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/image/undistorted",
                "object_detection_subscriber": "/object_detection/object",
                "sign_detection_subscriber": "/object_detection/sign",
                "crossing_detection_subscriber": "/crossing_detection/result",
                # Neues gebündeltes State-Topic
                "state_subscriber": "/tracking/state",
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
                "sign_detection_subscriber": (
                    Float32MultiArray,
                    self.sign_detection_callback,
                    1,
                ),
                "crossing_detection_subscriber": (
                    Float32MultiArray,
                    self.crossing_detection_callback,
                    1,
                ),
                # Lauschen auf das neue gebündelte State-Topic
                "state_subscriber": (
                    state_msgs.msg.State,
                    self.state_callback,
                    1,
                ),
            },
            published_topics={
                "debug_image_publisher": (Image, 1),
            },
        )
        self.cv_bridge = cv_bridge.CvBridge()

        # Track timeout handling (Neu: Basiert auf Detections statt State)
        self.last_obj_det_time = None
        self.last_sign_det_time = None
        self.last_cross_det_time = None
        self.track_timeout_sec = 0.5

        # Storage for latest messages
        self.latest_image = None
        self.latest_object_detections = []
        self.latest_sign_detections = []
        self.latest_crossing_detections = []

        self.latest_object_tracks = []
        self.latest_sign_tracks = []
        self.latest_crossing_tracks = []

        # Track timeout handling (nur noch ein Timer für den gesamten State)
        self.last_state_time = None
        self.track_timeout_sec = 0.5

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
            LANE_TYPE_EGO_SOLID: "Ego Solid",
            LANE_TYPE_EGO_DOTTED: "Ego Dotted",
            LANE_TYPE_OPP_SOLID: "Opp Solid",
            LANE_TYPE_OPP_DOTTED: "Opp Dotted",
            LANE_TYPE_RIGHT_SOLID: "Right Solid",
            LANE_TYPE_RIGHT_DOTTED: "Right Dotted",
            LANE_TYPE_LEFT_SOLID: "Left Solid",
            LANE_TYPE_LEFT_DOTTED: "Left Dotted",
        }

        self.crossing_colors = {
            LANE_TYPE_EGO_SOLID: "#00DD00",
            LANE_TYPE_EGO_DOTTED: "#00DD00",
            LANE_TYPE_OPP_SOLID: "#4488FF",
            LANE_TYPE_OPP_DOTTED: "#4488FF",
            LANE_TYPE_RIGHT_SOLID: "#FF8800",
            LANE_TYPE_RIGHT_DOTTED: "#FF8800",
            LANE_TYPE_LEFT_SOLID: "#FF44FF",
            LANE_TYPE_LEFT_DOTTED: "#FF44FF",
        }

        self.get_logger().info(
            "TrackingVisualizationNode initialized (State-Message based)"
        )

        self.last_frame_time = self.get_clock().now()
        self.fps = 0.0

        self.last_tracking_time = None
        self.tracking_fps = 0.0

        if COORD_TRANSFORM_AVAILABLE:
            try:
                self.coord_transform = CoordinateTransform(debug=False)
            except Exception as e:
                self.coord_transform = None
        else:
            self.coord_transform = None

    @property
    def show_velocity(self) -> bool:
        return self.get_parameter("show_velocity").value

    @property
    def text_size(self) -> int:
        return self.get_parameter("text_size").value

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def image_callback(self, msg: Image):
        try:
            if msg.encoding in ["mono8", "8UC1"]:
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width)
                )
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            self.latest_image = image

            if self.latest_image is not None:
                self.visualize_and_publish()

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def object_detection_callback(self, msg: Float32MultiArray):
        self.latest_object_detections = self.parse_detections(msg)
        self.last_obj_det_time = self.get_clock().now()

    def sign_detection_callback(self, msg: Float32MultiArray):
        self.latest_sign_detections = self.parse_detections(msg)
        self.last_sign_det_time = self.get_clock().now()

    def crossing_detection_callback(self, msg: Float32MultiArray):
        self.latest_crossing_detections = self.parse_detections(msg)
        self.last_cross_det_time = self.get_clock().now()

    def state_callback(self, msg: state_msgs.msg.State):
        """
        Callback for the new unified tracking state.
        Splits the tracked objects back into categories based on ID offsets.
        """
        current_time = self.get_clock().now()

        # Tracking FPS berechnen
        if self.last_tracking_time is not None:
            duration = (current_time - self.last_tracking_time).nanoseconds / 1e9
            if duration > 0:
                actual_fps = 1.0 / duration
                self.tracking_fps = (self.tracking_fps * 0.9) + (actual_fps * 0.1)
        self.last_tracking_time = current_time
        self.last_state_time = current_time

        obj_tracks = []
        sign_tracks = []
        cross_tracks = []

        for tracked_obj in msg.tracked_objects:
            # Zurück in ein Dictionary konvertieren (damit der Zeichen-Code gleich bleiben kann)
            track_dict = {
                "track_id": int(tracked_obj.tracked_id),
                "class_id": int(tracked_obj.class_id),
                "x": tracked_obj.position_x,
                "y": tracked_obj.position_y,
                "vx": tracked_obj.velocity_x,
                "vy": tracked_obj.velocity_y,
                "confidence": tracked_obj.confidence,
                "width": tracked_obj.width,
            }

            # Anhand der track_id entscheiden, in welche Liste es gehört
            tid = track_dict["track_id"]
            if tid < 10000:
                obj_tracks.append(track_dict)
            elif tid < 20000:
                sign_tracks.append(track_dict)
            else:
                cross_tracks.append(track_dict)

        self.latest_object_tracks = obj_tracks
        self.latest_sign_tracks = sign_tracks
        self.latest_crossing_tracks = cross_tracks

    # ------------------------------------------------------------------ #
    #  Message parsing (nur noch für Detections)                          #
    # ------------------------------------------------------------------ #

    def parse_detections(self, msg: Float32MultiArray):
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
                "center": {
                    "x": (data[i + 1] + data[i + 3]) / 2.0,
                    "y": (data[i + 2] + data[i + 4]) / 2.0,
                },
            }
            detections.append(detection)

        return detections

    # ------------------------------------------------------------------ #
    #  Die restlichen Helfer-Methoden bleiben IDENTISCH!                 #
    # ------------------------------------------------------------------ #

    def match_track_to_detection(self, track, detections):
        if not detections:
            return None
        best_match = None
        min_distance = float("inf")
        for detection in detections:
            if detection["class_id"] != track["class_id"]:
                continue
            dx = detection["center"]["x"] - track["x"]
            dy = detection["center"]["y"] - track["y"]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < min_distance and distance < 500:
                min_distance = distance
                best_match = detection
        return best_match

    def world_coords_to_bbox(self, track, image_shape):
        if self.coord_transform is not None:
            try:
                center_point = np.array([[track["x"], track["y"], 0]])
                center_pixel = self.coord_transform.world_to_camera(
                    center_point, input_unit=Unit.MILLIMETERS
                )[0]
                width_pixels = max(20, int(track["width"] * 0.1))
                height_pixels = int(width_pixels * 1.3)
                center_x, center_y = int(center_pixel[0]), int(center_pixel[1])
                return (
                    center_x - width_pixels // 2,
                    center_y - height_pixels // 2,
                    center_x + width_pixels // 2,
                    center_y + height_pixels // 2,
                )
            except Exception:
                pass
        return self._fallback_world_to_bbox(track, image_shape)

    def _fallback_world_to_bbox(self, track, image_shape):
        height, width = image_shape[:2]
        scale = 0.08
        center_x, center_y = width // 2, height - 80
        px, py = int(center_x + track["y"] * scale), int(center_y - track["x"] * scale)
        box_width = max(20, int(track["width"] * scale))
        box_height = int(box_width * 1.3)
        return (
            px - box_width // 2,
            py - box_height // 2,
            px + box_width // 2,
            py + box_height // 2,
        )

    def detection_to_bbox(self, detection, image_shape):
        if self.coord_transform is not None:
            try:
                bl = np.array(
                    [[detection["bottom_left"]["x"], detection["bottom_left"]["y"], 0]]
                )
                br = np.array(
                    [
                        [
                            detection["bottom_right"]["x"],
                            detection["bottom_right"]["y"],
                            0,
                        ]
                    ]
                )
                bl_pixel = self.coord_transform.world_to_camera(
                    bl, input_unit=Unit.MILLIMETERS
                )[0]
                br_pixel = self.coord_transform.world_to_camera(
                    br, input_unit=Unit.MILLIMETERS
                )[0]
                xmin, xmax = int(min(bl_pixel[0], br_pixel[0])), int(
                    max(bl_pixel[0], br_pixel[0])
                )
                height = int((xmax - xmin) * 1.3)
                ymax = int(max(bl_pixel[1], br_pixel[1]))
                return xmin, ymax - height, xmax, ymax
            except Exception:
                pass
        return self._fallback_detection_to_bbox(detection, image_shape)

    def _fallback_detection_to_bbox(self, detection, image_shape):
        height, width = image_shape[:2]
        scale = 0.08
        center_x, center_y = width // 2, height - 80
        bl_x = int(center_x + detection["bottom_left"]["y"] * scale)
        bl_y = int(center_y - detection["bottom_left"]["x"] * scale)
        br_x = int(center_x + detection["bottom_right"]["y"] * scale)
        br_y = int(center_y - detection["bottom_right"]["x"] * scale)
        xmin, xmax = min(bl_x, br_x), max(bl_x, br_x)
        ymin, ymax = min(bl_y, br_y) - int((xmax - xmin) * 1.2), max(bl_y, br_y)
        return xmin, ymin, xmax, ymax

    def detection_to_line_pixels(self, detection, bev=False):
        x1, y1 = detection["bottom_left"]["x"], detection["bottom_left"]["y"]
        x2, y2 = detection["bottom_right"]["x"], detection["bottom_right"]["y"]
        if bev:
            p1, p2 = self.bev_to_pixel(x1, y1), self.bev_to_pixel(x2, y2)
            if p1 is None or p2 is None:
                return None
        else:
            p1, p2 = self.world_to_pixel(x1, y1), self.world_to_pixel(x2, y2)
        return p1, p2

    def get_color_from_confidence(self, confidence):
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.6:
            return "yellow"
        elif confidence >= 0.4:
            return "orange"
        return "red"

    def get_crossing_color(self, class_id):
        return self.crossing_colors.get(class_id, "#FFFF00")

    def is_crossing_dotted(self, class_id):
        return class_id in (
            LANE_TYPE_EGO_DOTTED,
            LANE_TYPE_OPP_DOTTED,
            LANE_TYPE_RIGHT_DOTTED,
            LANE_TYPE_LEFT_DOTTED,
        )

    def world_to_pixel(self, x_world, y_world):
        if self.coord_transform is not None:
            try:
                point = np.array([[x_world, y_world, 0]])
                pixel = self.coord_transform.world_to_camera(
                    point, input_unit=Unit.MILLIMETERS
                )[0]
                return int(pixel[0]), int(pixel[1])
            except Exception:
                pass
        height, width = self.latest_image.shape[:2]
        return int(width // 2 + y_world * 0.08), int((height - 80) - x_world * 0.08)

    def bev_to_pixel(self, x_bev, y_bev):
        if self.coord_transform is not None:
            try:
                cam_point = self.coord_transform.bird_to_camera(
                    np.array([[x_bev, y_bev]])
                )[0]
                return int(cam_point[0]), int(cam_point[1])
            except Exception:
                pass
        return None

    def draw_track(
        self, draw, track, detections, is_sign, font_large, font_normal, coord_offset=0
    ):
        matched = self.match_track_to_detection(track, detections)
        if matched:
            xmin, ymin, xmax, ymax = self.detection_to_bbox(
                matched, self.latest_image.shape
            )
        else:
            xmin, ymin, xmax, ymax = self.world_coords_to_bbox(
                track, self.latest_image.shape
            )

        xmin += coord_offset
        xmax += coord_offset
        ymin += coord_offset
        ymax += coord_offset
        color = self.get_color_from_confidence(track["confidence"])

        if is_sign:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
            # Crash-Schutz: Inneres Rechteck nur zeichnen, wenn Platz dafür ist
            if (xmax - xmin) >= 6 and (ymax - ymin) >= 6:
                draw.rectangle(
                    [(xmin + 3, ymin + 3), (xmax - 3, ymax - 3)], outline=color, width=1
                )
        else:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)

        prefix = "S" if is_sign else "O"
        draw.text(
            (xmin + 5, ymin + 5),
            f"{prefix}#{track['track_id']}",
            fill=color,
            font=font_large,
        )
        draw.text(
            (xmin + 5, ymin + 25),
            self.class_names.get(track["class_id"], f"ID{track['class_id']}"),
            fill=color,
            font=font_normal,
        )
        draw.text(
            (xmin + 5, ymin + 40),
            f"{track['confidence']:.2f}",
            fill=color,
            font=font_normal,
        )

        if self.show_velocity and not is_sign:
            speed = np.sqrt(track["vx"] ** 2 + track["vy"] ** 2)
            if speed > 50:
                draw.text(
                    (xmin + 5, ymin + 55),
                    f"{int(speed)} mm/s",
                    fill=color,
                    font=font_normal,
                )
            cx, cy = self.world_to_pixel(track["x"], track["y"])
            fx, fy = self.world_to_pixel(
                track["x"] + track["vx"] * 1.0, track["y"] + track["vy"] * 1.0
            )
            cx += coord_offset
            cy += coord_offset
            fx += coord_offset
            fy += coord_offset
            draw.line([(cx, cy), (fx, fy)], fill="cyan", width=3)
            draw.ellipse([(fx - 4, fy - 4), (fx + 4, fy + 4)], fill="cyan")

    def draw_dashed_line(
        self, draw, p1, p2, color, width=3, dash_length=12, gap_length=8
    ):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            sx, sy = x1 + ux * pos, y1 + uy * pos
            end_pos = min(pos + dash_length, length)
            draw.line(
                [(int(sx), int(sy)), (int(x1 + ux * end_pos), int(y1 + uy * end_pos))],
                fill=color,
                width=width,
            )
            pos += dash_length + gap_length

    def draw_crossing_track(
        self, draw, track, detections, font_large, font_normal, coord_offset=0
    ):
        class_id = track["class_id"]
        color = self.get_crossing_color(class_id)
        is_dotted = self.is_crossing_dotted(class_id)

        matched = self.match_track_to_detection(track, detections)
        if matched:
            result = self.detection_to_line_pixels(matched, bev=True)
            if result is None:
                return
            p1, p2 = result
            p1 = (p1[0] + coord_offset, p1[1] + coord_offset)
            p2 = (p2[0] + coord_offset, p2[1] + coord_offset)
            if is_dotted:
                self.draw_dashed_line(draw, p1, p2, color=color, width=4)
            else:
                draw.line([p1, p2], fill=color, width=4)
            draw.ellipse([(p1[0] - 4, p1[1] - 4), (p1[0] + 4, p1[1] + 4)], fill=color)
            draw.ellipse([(p2[0] - 4, p2[1] - 4), (p2[0] + 4, p2[1] + 4)], fill=color)
            mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        else:
            pixel = self.bev_to_pixel(track["x"], track["y"])
            if pixel is None:
                return
            mid_x, mid_y = pixel[0] + coord_offset, pixel[1] + coord_offset
            diamond = [
                (mid_x, mid_y - 10),
                (mid_x + 10, mid_y),
                (mid_x, mid_y + 10),
                (mid_x - 10, mid_y),
            ]
            draw.polygon(diamond, outline=color, fill=None)
            if is_dotted:
                draw.ellipse(
                    [(mid_x - 3, mid_y - 3), (mid_x + 3, mid_y + 3)], fill=color
                )

        label_y = mid_y - 25
        draw.text(
            (mid_x + 5, label_y), f"C#{track['track_id']}", fill=color, font=font_large
        )
        draw.text(
            (mid_x + 5, label_y + 16),
            f"{self.class_names.get(class_id, f'Lane{class_id}')} ({track['confidence']:.2f})",
            fill=color,
            font=font_normal,
        )

    # ------------------------------------------------------------------ #
    #  Main visualization loop                                            #
    # ------------------------------------------------------------------ #

    def visualize_and_publish(self):
        if self.latest_image is None:
            return

        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_frame_time).nanoseconds / 1e9
        if time_diff > 0:
            self.fps = (self.fps * 0.9) + ((1.0 / time_diff) * 0.1)
        self.last_frame_time = current_time

        # Central timeout: clear all tracks if the state topic stops publishing
        if self.last_state_time is not None:
            if (
                (current_time - self.last_state_time).nanoseconds / 1e9
            ) > self.track_timeout_sec:
                self.latest_object_tracks = []
                self.latest_sign_tracks = []
                self.latest_crossing_tracks = []

        try:
            orig_height, orig_width = self.latest_image.shape[:2]
            padding = 200
            expanded_image = np.zeros(
                (orig_height + 2 * padding, orig_width + 2 * padding, 3), dtype=np.uint8
            )
            expanded_image[
                padding : padding + orig_height, padding : padding + orig_width
            ] = self.latest_image

            pil_image = PILImage.fromarray(expanded_image)
            draw = ImageDraw.Draw(pil_image)

            try:
                font_large = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    self.text_size + 8,
                )
                font_normal = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.text_size
                )
            except:
                font_large = font_normal = ImageFont.load_default()

            for track in self.latest_crossing_tracks:
                self.draw_crossing_track(
                    draw,
                    track,
                    self.latest_crossing_detections,
                    font_large,
                    font_normal,
                    padding,
                )
            for track in self.latest_object_tracks:
                self.draw_track(
                    draw,
                    track,
                    self.latest_object_detections,
                    False,
                    font_large,
                    font_normal,
                    padding,
                )
            for track in self.latest_sign_tracks:
                self.draw_track(
                    draw,
                    track,
                    self.latest_sign_detections,
                    True,
                    font_large,
                    font_normal,
                    padding,
                )

            stats_text = (
                f"Objects: {len(self.latest_object_tracks)} | Signs: {len(self.latest_sign_tracks)} | "
                f"Crossings: {len(self.latest_crossing_tracks)} | Frame: {self.frame_count} | "
                f"Vis-FPS: {self.fps:.1f} | Track FPS: {self.tracking_fps:.1f}"
            )
            draw.rectangle(
                [(5 + padding, 5 + padding), (560 + padding, 30 + padding)],
                fill="black",
                outline="white",
            )
            draw.text(
                (10 + padding, 10 + padding), stats_text, fill="white", font=font_normal
            )

            # Legende
            legend_y = 40 + padding
            draw.text(
                (10 + padding, legend_y),
                "O# = Object | S# = Sign | C# = Crossing Christian",
                fill="white",
                font=font_normal,
            )
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
            draw.line(
                [(130 + padding, legend_y + 6), (150 + padding, legend_y + 6)],
                fill=self.crossing_colors[LANE_TYPE_RIGHT_SOLID],
                width=2,
            )
            draw.text(
                (155 + padding, legend_y),
                "Right",
                fill=self.crossing_colors[LANE_TYPE_RIGHT_SOLID],
                font=font_normal,
            )
            draw.line(
                [(200 + padding, legend_y + 6), (220 + padding, legend_y + 6)],
                fill=self.crossing_colors[LANE_TYPE_LEFT_SOLID],
                width=2,
            )
            draw.text(
                (225 + padding, legend_y),
                "Left",
                fill=self.crossing_colors[LANE_TYPE_LEFT_SOLID],
                font=font_normal,
            )

            result_image = np.array(pil_image)[
                padding : padding + orig_height, padding : padding + orig_width
            ]
            self.debug_image_publisher.publish(
                self.cv_bridge.cv2_to_imgmsg(
                    cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), encoding="bgr8"
                )
            )
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Failed to visualize: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TrackingVisualizationNode()
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
