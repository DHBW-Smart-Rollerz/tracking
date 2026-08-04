#!/usr/bin/env python3
"""
Offline evaluation node: vergleicht Detection- oder Tracking-Output gegen CVAT-Ground-Truth.

Läuft als ROS2-Node während der Rosbag-Wiedergabe:

  Terminal 1 – Auswertung starten:
    ros2 run tracking evaluate_node --ros-args \
      -p annotations:=/path/to/annotations.xml \
      -p mode:=tracking \
      -p threshold_px:=50.0 \
      -p frame_step:=5

  Terminal 2 – Bag abspielen:
    ros2 bag play /path/to/rosbag --clock

Beim Beenden (Ctrl+C) werden die Ergebnisse ausgegeben.

Modus 'detection':  liest /object_detection/object + /object_detection/sign
Modus 'tracking':   liest /tracking/state
"""

import csv
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import state_msgs.msg

# Optional – nur im ROS2-Workspace verfügbar
try:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
        Unit,
    )

    COORD_TRANSFORM_AVAILABLE = True
except ImportError:
    COORD_TRANSFORM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Label → class_id Mapping (aus enums.py)
# ---------------------------------------------------------------------------

LABEL_TO_CLASS_ID: dict[str, int] = {
    "VEHICLE": 2,
    "PEDESTRIAN": 10,
    "STOP": 1,
    "NO_OVERTAKING": 3,
    "NO_OVERTAKING_LIFTED": 4,
    "FAST_TRACK": 5,
    "FAST_TRACK_LIFTED": 6,
    "SPEED_LIMIT_30": 7,
    "SPEED_LIMIT_30_LIFTED": 8,
    "CROSSWALK = 9": 9,  # CVAT-Labelname enthält " = 9"
    "PRIORITY_ONCOMING_TRAFFIC": 13,
    "PARKING": 14,
    "TURN_LEFT": 15,
    "TURN_RIGHT": 16,
    "GIVE_WAY": 17,
    "PRIORITY": 18,
    "PEDESTRIAN_ISLAND": 19,
}

CLASS_ID_TO_NAME: dict[int, str] = {v: k for k, v in LABEL_TO_CLASS_ID.items()}


# ---------------------------------------------------------------------------
# Annotation Parsing
# ---------------------------------------------------------------------------


def parse_annotations(xml_path: str) -> dict[int, list[tuple[int, float, float]]]:
    """
    Parsed CVAT-Annotations (Format 1.1, points).

    Returns:
        {annotation_id: [(class_id, pixel_x, pixel_y), ...]}
        Annotation-IDs ohne Objekte werden als leere Liste eingetragen.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt: dict[int, list] = {}
    skipped_labels: set[str] = set()

    for image_elem in root.findall("image"):
        ann_id = int(image_elem.get("id"))
        objects: list[tuple[int, float, float]] = []

        for points_elem in image_elem.findall("points"):
            label = points_elem.get("label", "")
            class_id = LABEL_TO_CLASS_ID.get(label)
            if class_id is None:
                skipped_labels.add(label)
                continue

            pts_str = points_elem.get("points", "")
            try:
                px, py = map(float, pts_str.split(","))
            except ValueError:
                continue  # Ungültiges Format, überspringen

            objects.append((class_id, px, py))

        gt[ann_id] = objects

    if skipped_labels:
        print(f"[evaluate_node] Übersprungene Labels (kein class_id): {skipped_labels}")

    return gt


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def greedy_match(
    predictions: list[tuple[int, float, float]],
    gt_objects: list[tuple[int, float, float]],
    threshold_px: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Greedy-Matching: findet das nächste Prediction-GT-Paar pro Klasse.
    Nur Paare mit gleicher class_id und Distanz <= threshold_px werden gematcht.

    Args:
        predictions: [(class_id, px, py), ...]
        gt_objects:  [(class_id, px, py), ...]
        threshold_px: maximale Pixeldistanz für einen Match

    Returns:
        matched_pairs:   [(pred_idx, gt_idx, distance), ...]
        unmatched_preds: [pred_idx, ...]   → False Positives
        unmatched_gts:   [gt_idx, ...]    → False Negatives
    """
    # Alle gültigen Paare berechnen (gleiche Klasse, innerhalb Schwelle)
    candidates: list[tuple[float, int, int]] = []
    for pi, (pc, ppx, ppy) in enumerate(predictions):
        for gi, (gc, gpx, gpy) in enumerate(gt_objects):
            if pc != gc:
                continue
            dist = math.sqrt((ppx - gpx) ** 2 + (ppy - gpy) ** 2)
            if dist <= threshold_px:
                candidates.append((dist, pi, gi))

    # Greedy: niedrigste Distanz zuerst
    candidates.sort()
    matched_pairs: list[tuple[int, int, float]] = []
    used_preds: set[int] = set()
    used_gts: set[int] = set()

    for dist, pi, gi in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        matched_pairs.append((pi, gi, dist))
        used_preds.add(pi)
        used_gts.add(gi)

    unmatched_preds = [i for i in range(len(predictions)) if i not in used_preds]
    unmatched_gts = [i for i in range(len(gt_objects)) if i not in used_gts]

    return matched_pairs, unmatched_preds, unmatched_gts


# ---------------------------------------------------------------------------
# ROS2 Evaluation Node
# ---------------------------------------------------------------------------


class EvaluationNode(Node):
    """
    Wertet Detection- oder Tracking-Output gegen CVAT-Ground-Truth aus.

    Zählt Frames anhand des Image-Topics (kein Timestamp-Parsing nötig).
    Bei jedem annotierten Frame (frame_idx % frame_step == 0) wird die
    aktuellste Detection/Tracking-Message mit dem GT verglichen.
    """

    def __init__(self):
        super().__init__("evaluation_node")

        # ---- Parameter ----
        self.declare_parameter("annotations", "")
        self.declare_parameter("mode", "tracking")  # 'detection' | 'tracking'
        self.declare_parameter("frame_step", 5)  # jeder N-te Frame wurde gelabelt
        self.declare_parameter("threshold_px", 50.0)  # Match-Distanz in Pixeln
        self.declare_parameter("image_topic", "/camera/image/undistorted")
        self.declare_parameter("object_det_topic", "/object_detection/object")
        self.declare_parameter("sign_det_topic", "/object_detection/sign")
        self.declare_parameter("tracking_topic", "/tracking/state")
        self.declare_parameter(
            "output_dir", "~/smarty_workspace/src/tracking/evaluation_results2"
        )

        ann_path = self.get_parameter("annotations").value
        self.mode = self.get_parameter("mode").value
        self.frame_step = self.get_parameter("frame_step").value
        self.threshold_px = self.get_parameter("threshold_px").value

        if not ann_path:
            self.get_logger().error(
                "Parameter 'annotations' fehlt! "
                "Bitte -p annotations:=/pfad/zur/annotations.xml angeben."
            )
            raise RuntimeError("annotations parameter required")

        if self.mode not in ("detection", "tracking"):
            raise ValueError(
                f"mode muss 'detection' oder 'tracking' sein, nicht '{self.mode}'"
            )

        # ---- Ground Truth laden ----
        self.gt = parse_annotations(ann_path)
        total_gt_objects = sum(len(v) for v in self.gt.values())
        self.get_logger().info(
            f"Ground Truth: {total_gt_objects} Objekte in {len(self.gt)} Frames geladen"
        )

        # ---- Koordinatentransformation ----
        self.coord_transform = None
        if COORD_TRANSFORM_AVAILABLE:
            try:
                self.coord_transform = CoordinateTransform(debug=False)
                self.get_logger().info("CoordinateTransform erfolgreich geladen")
            except Exception as e:
                self.get_logger().warn(
                    f"CoordinateTransform-Init fehlgeschlagen: {e} → Fallback aktiv"
                )
        else:
            self.get_logger().warn(
                "camera_preprocessing nicht importierbar → linearer Fallback für Koordinaten"
            )

        # ---- Interner Zustand ----
        self.image_counter: int = 0
        self.image_width: int = 800  # Default, wird aus Image-Message aktualisiert
        self.image_height: int = 640

        # Zwischengespeicherte Predictions (Weltkoordinaten in mm)
        # Format: [(class_id, world_x, world_y), ...]
        self.latest_obj_detections: list[tuple[int, float, float]] = []
        self.latest_sign_detections: list[tuple[int, float, float]] = []
        self.latest_tracking: list[tuple[int, float, float]] = []

        # Metriken: {class_id: {tp, fp, fn, distances}}
        self.metrics: dict[int, dict] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "distances": []}
        )
        self.evaluated_frames: int = 0
        self.skipped_frames: int = 0  # GT vorhanden, aber noch keine Predictions

        # ---- Subscriptions ----
        image_topic = self.get_parameter("image_topic").value
        self.create_subscription(Image, image_topic, self._image_cb, 1)

        if self.mode == "detection":
            obj_topic = self.get_parameter("object_det_topic").value
            sign_topic = self.get_parameter("sign_det_topic").value
            self.create_subscription(Float32MultiArray, obj_topic, self._obj_det_cb, 10)
            self.create_subscription(
                Float32MultiArray, sign_topic, self._sign_det_cb, 10
            )
            self.get_logger().info(
                f"Modus DETECTION | Topics: {obj_topic}, {sign_topic}"
            )
        else:
            track_topic = self.get_parameter("tracking_topic").value
            self.create_subscription(
                state_msgs.msg.State, track_topic, self._tracking_cb, 10
            )
            self.get_logger().info(f"Modus TRACKING | Topic: {track_topic}")

        self.get_logger().info(
            f"Bereit | frame_step={self.frame_step}, threshold={self.threshold_px:.0f}px"
        )

    # -------------------------------------------------------------------------
    # Detection / Tracking Callbacks
    # -------------------------------------------------------------------------

    def _obj_det_cb(self, msg: Float32MultiArray):
        self.latest_obj_detections = self._parse_float_array(msg)

    def _sign_det_cb(self, msg: Float32MultiArray):
        self.latest_sign_detections = self._parse_float_array(msg)

    def _tracking_cb(self, msg: state_msgs.msg.State):
        self.latest_tracking = [
            (int(t.class_id), float(t.position_x), float(t.position_y))
            for t in msg.tracked_objects
        ]

    # -------------------------------------------------------------------------
    # Image Callback – Kern der Auswertung
    # -------------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        """Zählt jeden eingehenden Frame und wertet annotierte Frames aus."""
        self.image_width = msg.width
        self.image_height = msg.height

        frame_idx = self.image_counter
        self.image_counter += 1

        # Nur annotierte Frames auswerten
        if frame_idx % self.frame_step != 0:
            return

        ann_id = frame_idx // self.frame_step
        if ann_id not in self.gt:
            return  # Außerhalb des annotierten Bereichs

        gt_objects = self.gt[ann_id]

        # Aktuelle Predictions sammeln
        if self.mode == "detection":
            preds_world = self.latest_obj_detections + self.latest_sign_detections
        else:
            preds_world = self.latest_tracking

        # Noch keine Predictions empfangen → Frame überspringen
        if not preds_world and not gt_objects:
            return
        if not preds_world and gt_objects:
            # Keine Predictions, aber GT vorhanden → alles FN
            # Trotzdem auswerten, damit FN gezählt werden
            pass

        # Weltkoordinaten → Pixel
        preds_px: list[tuple[int, float, float]] = []
        for class_id, wx, wy in preds_world:
            px, py = self._world_to_pixel(wx, wy)
            preds_px.append((class_id, px, py))

        # FOV-Filter: Tracks außerhalb des Kamerabildes ausschließen.
        # Der KF verfolgt Objekte auch wenn sie nicht sichtbar sind – diese
        # können nicht in der GT auftauchen und würden sonst als FP gezählt.
        preds_px = [
            (cid, px, py)
            for (cid, px, py) in preds_px
            if 0.0 <= px < self.image_width and 0.0 <= py < self.image_height
        ]

        # Matching
        matched, unmatched_preds, unmatched_gts = greedy_match(
            preds_px, gt_objects, self.threshold_px
        )

        # Metriken akkumulieren
        for pi, gi, dist in matched:
            cid = preds_px[pi][0]
            self.metrics[cid]["tp"] += 1
            self.metrics[cid]["distances"].append(dist)

        for pi in unmatched_preds:
            cid = preds_px[pi][0]
            self.metrics[cid]["fp"] += 1

        for gi in unmatched_gts:
            cid = gt_objects[gi][0]
            self.metrics[cid]["fn"] += 1

        self.evaluated_frames += 1

        if self.evaluated_frames % 20 == 0:
            self.get_logger().info(
                f"Fortschritt: {self.evaluated_frames} Frames ausgewertet "
                f"(ann_id={ann_id}, frame_idx={frame_idx})"
            )

    # -------------------------------------------------------------------------
    # Hilfsmethoden
    # -------------------------------------------------------------------------

    def _parse_float_array(
        self, msg: Float32MultiArray
    ) -> list[tuple[int, float, float]]:
        """
        Parsed Float32MultiArray-Detection-Nachricht.
        Format: [class_id, bl_x, bl_y, br_x, br_y, score] je Detection.
        Gibt den unteren Mittelpunkt zurück (Mittelpunkt zwischen bl und br).
        """
        data = msg.data
        if not data or len(data) % 6 != 0:
            return []

        results = []
        for i in range(0, len(data), 6):
            class_id = int(data[i])
            bl_x, bl_y = float(data[i + 1]), float(data[i + 2])
            br_x, br_y = float(data[i + 3]), float(data[i + 4])
            center_x = (bl_x + br_x) / 2.0
            center_y = (bl_y + br_y) / 2.0
            results.append((class_id, center_x, center_y))
        return results

    def _world_to_pixel(self, x_world: float, y_world: float) -> tuple[float, float]:
        """
        Wandelt Weltkoordinaten (mm) in Pixelkoordinaten um.
        Verwendet CoordinateTransform falls verfügbar, sonst linearen Fallback.
        """
        if self.coord_transform is not None:
            try:
                point = np.array([[x_world, y_world, 0.0]])
                pixel = self.coord_transform.world_to_camera(
                    point, input_unit=Unit.MILLIMETERS
                )[0]
                return float(pixel[0]), float(pixel[1])
            except Exception:
                pass  # Fallback

        # Linearer Fallback (identisch mit TrackingVisualizationNode)
        cx = self.image_width // 2
        cy = self.image_height - 80
        return float(cx + y_world * 0.08), float(cy - x_world * 0.08)

    # -------------------------------------------------------------------------
    # Ergebnisausgabe
    # -------------------------------------------------------------------------

    def _compute_results(self) -> tuple[list[dict], dict]:
        """
        Berechnet per-Klasse- und Gesamt-Metriken aus self.metrics.

        Returns:
            per_class: Liste von Dicts (eine Zeile pro Klasse)
            totals:    Dict mit aggregierten Gesamtwerten
        """
        per_class: list[dict] = []
        total_tp = total_fp = total_fn = 0
        all_distances: list[float] = []

        for cid in sorted(self.metrics.keys()):
            m = self.metrics[cid]
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            dists = m["distances"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            mean_err = sum(dists) / len(dists) if dists else None

            per_class.append(
                {
                    "class_id": cid,
                    "class_name": CLASS_ID_TO_NAME.get(cid, f"class_{cid}"),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "mean_err_px": round(mean_err, 2) if mean_err is not None else None,
                }
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_distances.extend(dists)

        total_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        total_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        total_f1 = (
            2.0 * total_precision * total_recall / (total_precision + total_recall)
            if (total_precision + total_recall) > 0
            else 0.0
        )
        total_mean_err = (
            sum(all_distances) / len(all_distances) if all_distances else None
        )

        totals = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(total_precision, 4),
            "recall": round(total_recall, 4),
            "f1": round(total_f1, 4),
            "mean_err_px": (
                round(total_mean_err, 2) if total_mean_err is not None else None
            ),
        }
        return per_class, totals

    def print_results(self):
        """Gibt die vollständige Auswertung auf der Konsole aus und speichert CSV + JSON."""
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  AUSWERTUNGSERGEBNIS")
        print(f"  Modus:               {self.mode.upper()}")
        print(f"  Match-Schwelle:      {self.threshold_px:.0f} px")
        print(f"  Frame-Step:          jeder {self.frame_step}. Frame")
        print(f"  Ausgewertete Frames: {self.evaluated_frames}")
        print(sep)

        if self.evaluated_frames == 0:
            print(
                "  WARNUNG: Keine Frames ausgewertet. Topics korrekt? Bag vollständig abgespielt?"
            )
            print(sep)
            return

        per_class, totals = self._compute_results()

        # ---- Konsole ----
        print(
            f"  {'Klasse':<28} {'TP':>5} {'FP':>5} {'FN':>5} "
            f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'Err[px]':>9}"
        )
        print("-" * 72)
        for row in per_class:
            err_str = (
                f"{row['mean_err_px']:>9.1f}"
                if row["mean_err_px"] is not None
                else f"{'—':>9}"
            )
            print(
                f"  {row['class_name']:<28} {row['tp']:>5} {row['fp']:>5} {row['fn']:>5} "
                f"{row['precision']:>7.3f} {row['recall']:>7.3f} {row['f1']:>7.3f} {err_str}"
            )

        print("-" * 72)
        total_err_str = (
            f"{totals['mean_err_px']:>9.1f}"
            if totals["mean_err_px"] is not None
            else f"{'—':>9}"
        )
        print(
            f"  {'GESAMT':<28} {totals['tp']:>5} {totals['fp']:>5} {totals['fn']:>5} "
            f"{totals['precision']:>7.3f} {totals['recall']:>7.3f} {totals['f1']:>7.3f} {total_err_str}"
        )
        print(sep)
        print("  F1      = Harmonisches Mittel aus Precision und Recall")
        print("  Err[px] = Mittlere Pixeldistanz der gematchten Paare")
        print(
            f"  TP={totals['tp']}, FP={totals['fp']} (Geister), FN={totals['fn']} (übersehen)"
        )
        print(sep + "\n")

        # ---- Dateien speichern ----
        self._save_results(per_class, totals)

    def _save_results(self, per_class: list[dict], totals: dict):
        """Speichert Ergebnisse als CSV und JSON in output_dir."""
        output_dir = os.path.expanduser(self.get_parameter("output_dir").value)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        basename = f"eval_{self.mode}_{timestamp}"

        # ---- CSV (per Klasse + Gesamtzeile) ----
        csv_path = os.path.join(output_dir, f"{basename}.csv")
        fieldnames = [
            "class_id",
            "class_name",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_err_px",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_class)
            # Gesamtzeile
            writer.writerow(
                {
                    "class_id": "TOTAL",
                    "class_name": "GESAMT",
                    **totals,
                }
            )
        print(f"  CSV gespeichert:  {csv_path}")

        # ---- JSON (vollständig mit Metadaten) ----
        json_path = os.path.join(output_dir, f"{basename}.json")
        result_doc = {
            "meta": {
                "mode": self.mode,
                "threshold_px": self.threshold_px,
                "frame_step": self.frame_step,
                "evaluated_frames": self.evaluated_frames,
                "timestamp": timestamp,
            },
            "per_class": per_class,
            "total": totals,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_doc, f, indent=2, ensure_ascii=False)
        print(f"  JSON gespeichert: {json_path}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_results()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
