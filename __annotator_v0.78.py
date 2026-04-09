import csv
from datetime import datetime
import os
import random
import re
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial

try:
    import numpy as np
except Exception as exc:
    print("Missing dependency: numpy. Install with: pip install numpy")
    raise SystemExit(1) from exc

try:
    import cv2
except Exception as exc:
    print("Missing dependency: opencv-python. Install with: pip install opencv-python")
    raise SystemExit(1) from exc

try:
    from PyQt5.QtCore import QByteArray, QEvent, QPoint, QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QCursor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QAction,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QKeySequenceEdit,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressDialog,
        QShortcut,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:
    print("Missing dependency: PyQt5. Install with: pip install PyQt5")
    raise SystemExit(1) from exc

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None



IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_FILE_NAMES = ("classes.txt", "obj.names")
CLASS_NAME_ALIASES = {
    "padestrian": "pedestrian",
    "pedestrain": "pedestrian",
    "person": "pedestrian",
    "people": "pedestrian",
    "autorickshaw": "auto",
    "auto_rickshaw": "auto",
    "threewheeler": "auto",
    "threewheel": "auto",
    "bike": "motorcycle",
    "motorbike": "motorcycle",
}
HANDLE_SIZE = 6
MIN_BOX_SIZE = 4
REVIEW_SUFFIX = ".reviewed"
SETTINGS_ORG = "Arnob"
SETTINGS_APP = "YOLOAdvancedAnnotator"
CRASH_LOG_NAME = "annotator_crash.log"
MAX_FEATURE_CACHE = 64
LEGACY_DEFAULT_SHORTCUTS = {
    "undo": "Ctrl+Z",
    "reset": "Ctrl+Shift+R",
    "prev_image_alt": "PgUp",
    "next_image_alt": "PgDown",
    "nudge_left": "Left",
    "nudge_right": "Right",
    "nudge_up": "Up",
    "nudge_down": "Down",
    "nudge_left_fast": "Shift+Left",
    "nudge_right_fast": "Shift+Right",
    "nudge_up_fast": "Shift+Up",
    "nudge_down_fast": "Shift+Down",
}
DEFAULT_SHORTCUTS = {
    "select_all_annotations": "Ctrl+A",
    "open_folder": "Ctrl+O",
    "save": "Ctrl+S",
    "load_model": "Ctrl+M",
    "copy": "Ctrl+C",
    "cut": "Ctrl+X",
    "paste": "Ctrl+V",
    "duplicate": "Ctrl+D",
    "copy_prev_labels": "Ctrl+P",
    "copy_to_next_suggest": "Ctrl+Shift+N",
    "toggle_auto_next_suggest": "",
    "undo": "Z",
    "redo": "Y",
    "reset": "C",
    "accept_suggestion": "Return",
    "reject_suggestion": "Backspace",
    "accept_all_suggestions": "Ctrl+Return",
    "accept_roi_suggestions": "Ctrl+Alt+Return",
    "reject_suggestions": "Shift+Backspace",
    "reject_suggestions_only": "X",
    "delete": "Delete",
    "set_selected_class": "L",
    "auto_current": "Q",
    "clear_roi": "Esc",
    "toggle_roi_mode": "G",
    "delete_roi": "Ctrl+Delete",
    "set_roi_class": "Ctrl+L",
    "fit": "F",
    "actual_size": "0",
    "prev_image": "A",
    "next_image": "D",
    "next_unlabeled": "W",
    "next_unlabeled_alt": "N",
    "prev_image_alt": "Alt+Left",
    "next_image_alt": "Alt+Right",
    "nudge_left": "Ctrl+Left",
    "nudge_right": "Ctrl+Right",
    "nudge_up": "Ctrl+Up",
    "nudge_down": "Ctrl+Down",
    "nudge_left_fast": "Ctrl+Shift+Left",
    "nudge_right_fast": "Ctrl+Shift+Right",
    "nudge_up_fast": "Ctrl+Shift+Up",
    "nudge_down_fast": "Ctrl+Shift+Down",
    "pan_left": "Space+Left",
    "pan_right": "Space+Right",
    "pan_up": "Space+Up",
    "pan_down": "Space+Down",
    "zoom_in": "Plus",
    "zoom_out": "Minus",
    "toggle_reviewed": "R",
    "next_suggestion": "Tab",
    "prev_suggestion": "Shift+Tab",
    "tab_project": "Ctrl+1",
    "tab_review": "Ctrl+2",
}


@dataclass
class Annotation:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    score: float = -1.0

    def normalized(self):
        left = min(self.x1, self.x2)
        right = max(self.x1, self.x2)
        top = min(self.y1, self.y2)
        bottom = max(self.y1, self.y2)
        return Annotation(left, top, right, bottom, self.class_id, self.score)

    def width(self):
        box = self.normalized()
        return box.x2 - box.x1

    def height(self):
        box = self.normalized()
        return box.y2 - box.y1

    def contains(self, x, y):
        box = self.normalized()
        return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2

    def move_by(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy

    def clamp(self, width, height):
        box = self.normalized()
        box.x1 = max(0, min(box.x1, width))
        box.x2 = max(0, min(box.x2, width))
        box.y1 = max(0, min(box.y1, height))
        box.y2 = max(0, min(box.y2, height))
        if box.width() >= MIN_BOX_SIZE and box.height() >= MIN_BOX_SIZE:
            self.x1, self.y1, self.x2, self.y2 = box.x1, box.y1, box.x2, box.y2

    def to_yolo(self, width, height):
        box = self.normalized()
        xc = ((box.x1 + box.x2) / 2.0) / width
        yc = ((box.y1 + box.y2) / 2.0) / height
        bw = box.width() / width
        bh = box.height() / height
        return self.class_id, xc, yc, bw, bh

    @staticmethod
    def from_yolo(class_id, xc, yc, bw, bh, width, height):
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        return Annotation(x1, y1, x2, y2, int(class_id), -1.0).normalized()


class ShortcutDialog(QDialog):
    def __init__(self, shortcuts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Shortcuts")
        self.resize(520, 640)
        self.editors = {}

        layout = QVBoxLayout(self)
        form_host = QWidget()
        form = QFormLayout(form_host)
        for key in sorted(shortcuts):
            editor = QKeySequenceEdit(QKeySequence(shortcuts[key]))
            self.editors[key] = editor
            field = QWidget()
            field_layout = QHBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(6)
            clear_btn = QPushButton("Clear")
            clear_btn.setFixedWidth(60)
            clear_btn.clicked.connect(lambda _checked=False, target=editor: target.clear())
            field_layout.addWidget(editor, 1)
            field_layout.addWidget(clear_btn)
            form.addRow(key.replace("_", " ").title(), field)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        return {key: editor.keySequence().toString() for key, editor in self.editors.items()}

    def accept(self):
        values = self.values()
        seen = {}
        conflicts = []
        for key, value in values.items():
            sequence = QKeySequence(value)
            if sequence.isEmpty():
                continue
            for index in range(sequence.count()):
                normalized = QKeySequence(sequence[index]).toString(QKeySequence.NativeText)
                if not normalized:
                    continue
                if normalized in seen:
                    conflicts.append((normalized, seen[normalized], key))
                else:
                    seen[normalized] = key
        if conflicts:
            lines = ["These shortcuts conflict:"]
            for sequence, first_key, second_key in conflicts[:12]:
                lines.append(
                    f"- {sequence}: {first_key.replace('_', ' ')} / {second_key.replace('_', ' ')}"
                )
            QMessageBox.warning(self, "Shortcut Conflict", "\n".join(lines))
            return
        super().accept()


def atomic_write_lines(path, lines):
    directory = os.path.dirname(path) or "."
    fd, temp_path = tempfile.mkstemp(prefix=".annotator_", suffix=".tmp", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def encode_state_bytes(value):
    if value is None:
        return ""
    try:
        return bytes(value.toBase64()).decode("ascii")
    except Exception:
        return ""


def decode_state_bytes(value):
    if not value:
        return None
    try:
        return QByteArray.fromBase64(value.encode("ascii"))
    except Exception:
        return None


class Canvas(QWidget):
    annotations_changed = pyqtSignal()
    selection_changed = pyqtSignal(int)
    mouse_position_changed = pyqtSignal(str)
    suggestions_changed = pyqtSignal()
    roi_changed = pyqtSignal(str)
    class_edit_requested = pyqtSignal(int)
    navigate_requested = pyqtSignal(int)
    navigate_requested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.image = None
        self.image_rgb = None
        self.pixmap = None
        self.image_path = None

        self.classes = []
        self.annotations = []
        self.selected_index = -1
        self.selected_indices = set()
        self.current_class_id = 0

        self.scale = 1.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.base_offset_x = 0.0
        self.base_offset_y = 0.0

        self.mode = "idle"
        self.active_handle = None
        self.drag_start_image = None
        self.drag_start_box = None
        self.drag_start_boxes = None
        self.temp_box = None
        self.last_mouse_pos = QPoint()
        self.clipboard_annotation = None
        self.clipboard_annotations = []
        self.show_boxes = True
        self.space_pressed = False
        self.force_new_box_mode = False
        self.history = []
        self.redo_history = []
        self.max_history = 200
        self.suggestions = []
        self.selected_suggestion_index = -1
        self.roi_points = []
        self.roi_preview_point = None
        self.roi_mode = False
        self.ui_font_size = 10
        self.plus_cursor = self.build_plus_cursor()

    def build_plus_cursor(self):
        size = 19
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        pen = QPen(QColor(30, 30, 30), 2)
        painter.setPen(pen)
        center = size // 2
        painter.drawLine(center, 2, center, size - 3)
        painter.drawLine(2, center, size - 3, center)
        painter.end()
        return QCursor(pixmap, center, center)

    def set_ui_font_size(self, point_size):
        self.ui_font_size = max(8, min(int(point_size), 24))
        self.update()

    def has_image(self):
        return self.image is not None

    def clone_annotations(self, annotations):
        return [Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score) for item in annotations]

    def snapshot_state(self):
        return {
            "annotations": self.clone_annotations(self.annotations),
            "suggestions": self.clone_annotations(self.suggestions),
            "selected_index": self.selected_index,
            "selected_indices": set(self.selected_indices),
            "selected_suggestion_index": self.selected_suggestion_index,
        }

    def restore_state(self, state):
        self.annotations = self.clone_annotations(state.get("annotations", []))
        self.suggestions = self.clone_annotations(state.get("suggestions", []))
        self.selected_indices = set(state.get("selected_indices", set()))
        self.selected_index = -1
        restored_suggestion_index = state.get("selected_suggestion_index", -1)
        self.selected_suggestion_index = min(max(restored_suggestion_index, -1), len(self.suggestions) - 1)
        self.set_selected_index(state.get("selected_index", -1))
        self.suggestions_changed.emit()
        self.annotations_changed.emit()
        self.update()

    def push_history(self):
        self.history.append(self.snapshot_state())
        self.redo_history = []
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            return
        self.redo_history.append(self.snapshot_state())
        if len(self.redo_history) > self.max_history:
            self.redo_history.pop(0)
        self.restore_state(self.history.pop())

    def redo(self):
        if not self.redo_history:
            return
        self.history.append(self.snapshot_state())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.restore_state(self.redo_history.pop())

    def reset_annotations(self):
        if not self.annotations:
            return
        self.push_history()
        self.annotations = []
        self.set_selected_index(-1)
        self.annotations_changed.emit()
        self.update()

    def clear_suggestions(self):
        self.suggestions = []
        self.selected_suggestion_index = -1
        self.selection_changed.emit(self.selected_index)
        self.suggestions_changed.emit()
        self.update()

    def set_suggestions(self, suggestions):
        self.suggestions = suggestions
        self.selected_suggestion_index = 0 if suggestions else -1
        self.suggestions_changed.emit()
        self.update()

    def accept_selected_suggestion(self):
        if not (0 <= self.selected_suggestion_index < len(self.suggestions)):
            return
        self.push_history()
        suggestion = self.suggestions.pop(self.selected_suggestion_index)
        self.annotations.append(suggestion)
        self.set_selected_index(len(self.annotations) - 1)
        if self.selected_suggestion_index >= len(self.suggestions):
            self.selected_suggestion_index = len(self.suggestions) - 1
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()

    def reject_selected_suggestion(self):
        if not (0 <= self.selected_suggestion_index < len(self.suggestions)):
            return
        self.push_history()
        self.suggestions.pop(self.selected_suggestion_index)
        if self.selected_suggestion_index >= len(self.suggestions):
            self.selected_suggestion_index = len(self.suggestions) - 1
        self.suggestions_changed.emit()
        self.update()

    def accept_all_suggestions(self):
        if not self.suggestions:
            return
        self.push_history()
        self.annotations.extend(self.suggestions)
        self.suggestions = []
        self.selected_suggestion_index = -1
        self.set_selected_index(len(self.annotations) - 1 if self.annotations else -1)
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()

    def reject_all_suggestions(self):
        if not self.suggestions:
            return
        self.push_history()
        self.clear_suggestions()

    def cycle_suggestion(self, direction):
        if not self.suggestions:
            self.selected_suggestion_index = -1
        elif self.selected_suggestion_index < 0:
            self.selected_suggestion_index = 0
        else:
            self.selected_suggestion_index = (self.selected_suggestion_index + direction) % len(self.suggestions)
        self.suggestions_changed.emit()
        self.update()

    def set_roi_mode(self, enabled):
        self.roi_mode = enabled
        if not enabled:
            self.roi_points = []
            self.roi_preview_point = None
            self.roi_changed.emit("")
        self.update()

    def has_roi(self):
        return len(self.roi_points) >= 3

    def clear_roi(self):
        self.roi_points = []
        self.roi_preview_point = None
        self.roi_mode = False
        self.roi_changed.emit("")
        self.update()

    def roi_bounds_text(self):
        if not self.has_roi():
            return ""
        xs = [point[0] for point in self.roi_points]
        ys = [point[1] for point in self.roi_points]
        return f"ROI polygon: ({int(min(xs))}, {int(min(ys))}) - ({int(max(xs))}, {int(max(ys))})"

    def point_in_polygon(self, x, y, polygon_points):
        if len(polygon_points) < 3:
            return True
        inside = False
        j = len(polygon_points) - 1
        for i in range(len(polygon_points)):
            xi, yi = polygon_points[i]
            xj, yj = polygon_points[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def inside_roi(self, annotation, roi_points=None):
        if not roi_points:
            return True
        box = annotation.normalized()
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0
        return self.point_in_polygon(cx, cy, roi_points)

    def collect_annotations_in_roi(self):
        if not self.has_roi():
            return []
        return [index for index, annotation in enumerate(self.annotations) if self.inside_roi(annotation, self.roi_points)]

    def collect_suggestions_in_roi(self):
        if not self.has_roi():
            return []
        return [index for index, suggestion in enumerate(self.suggestions) if self.inside_roi(suggestion, self.roi_points)]

    def delete_suggestions_in_roi(self):
        if not self.has_roi() or not self.suggestions:
            return 0
        affected = self.collect_suggestions_in_roi()
        if not affected:
            return 0
        self.push_history()
        self.suggestions = [item for index, item in enumerate(self.suggestions) if index not in set(affected)]
        self.selected_suggestion_index = 0 if self.suggestions else -1
        self.suggestions_changed.emit()
        self.update()
        return len(affected)

    def accept_suggestions_in_roi(self):
        if not self.has_roi() or not self.suggestions:
            return 0
        affected = self.collect_suggestions_in_roi()
        if not affected:
            return 0
        self.push_history()
        accepted = [self.suggestions[index] for index in affected]
        self.annotations.extend(accepted)
        self.suggestions = [item for index, item in enumerate(self.suggestions) if index not in set(affected)]
        self.selected_suggestion_index = 0 if self.suggestions else -1
        self.set_selected_index(len(self.annotations) - 1 if self.annotations else -1)
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()
        return len(accepted)

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read image: {path}")

        self.image_path = path
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = self.image.shape[:2]
        self.pixmap = QPixmap.fromImage(QImage(self.image_rgb.data, w, h, 3 * w, QImage.Format_RGB888))
        self.annotations = []
        self.selected_index = -1
        self.temp_box = None
        self.mode = "idle"
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.history = []
        self.redo_history = []
        self.clear_suggestions()
        self.roi_points = []
        self.roi_preview_point = None
        self.load_labels()
        self.fit_to_window()
        self.annotations_changed.emit()
        self.selection_changed.emit(self.selected_index)
        self.update()

    def clear_loaded_image(self):
        self.image = None
        self.image_rgb = None
        self.pixmap = None
        self.image_path = None
        self.annotations = []
        self.suggestions = []
        self.selected_index = -1
        self.selected_indices = set()
        self.selected_suggestion_index = -1
        self.temp_box = None
        self.mode = "idle"
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.drag_start_image = None
        self.drag_start_box = None
        self.drag_start_boxes = None
        self.history = []
        self.redo_history = []
        self.roi_points = []
        self.roi_preview_point = None
        self.clipboard_annotation = None
        self.clipboard_annotations = []
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.selection_changed.emit(-1)
        self.update()

    def fit_to_window(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def actual_size(self):
        if not self.has_image():
            return
        self.zoom = 1.0 / max(self.compute_base_scale(), 0.001)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def compute_base_scale(self):
        if not self.has_image():
            return 1.0
        h, w = self.image.shape[:2]
        return min(self.width() / max(w, 1), self.height() / max(h, 1))

    def current_scale(self):
        return self.scale * self.zoom

    def update_view_metrics(self):
        if not self.has_image():
            self.scale = 1.0
            self.base_offset_x = 0.0
            self.base_offset_y = 0.0
            return
        h, w = self.image.shape[:2]
        self.scale = self.compute_base_scale()
        scaled_w = w * self.current_scale()
        scaled_h = h * self.current_scale()
        self.base_offset_x = (self.width() - scaled_w) / 2.0 + self.pan_x
        self.base_offset_y = (self.height() - scaled_h) / 2.0 + self.pan_y

    def image_to_display(self, x, y):
        scale = self.current_scale()
        return x * scale + self.base_offset_x, y * scale + self.base_offset_y

    def display_to_image(self, x, y):
        scale = self.current_scale()
        if scale == 0:
            return 0.0, 0.0
        return (x - self.base_offset_x) / scale, (y - self.base_offset_y) / scale

    def clamp_point(self, x, y):
        if not self.has_image():
            return x, y
        h, w = self.image.shape[:2]
        return max(0, min(x, w)), max(0, min(y, h))

    def load_labels(self):
        self.annotations = []
        label_path = self.label_path()
        if not os.path.exists(label_path):
            return
        h, w = self.image.shape[:2]
        with open(label_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id, xc, yc, bw, bh = map(float, parts)
                except ValueError:
                    continue
                if bw <= 0 or bh <= 0:
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    continue
                self.annotations.append(Annotation.from_yolo(class_id, xc, yc, bw, bh, w, h))

    def save_labels(self):
        if not self.has_image():
            return
        label_path = self.label_path()
        if not self.annotations:
            if os.path.exists(label_path):
                os.remove(label_path)
            return
        h, w = self.image.shape[:2]
        lines = []
        for annotation in self.annotations:
            class_id, xc, yc, bw, bh = annotation.to_yolo(w, h)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        atomic_write_lines(label_path, lines)

    def label_path(self):
        base, _ = os.path.splitext(self.image_path)
        return base + ".txt"

    def set_classes(self, classes):
        self.classes = classes
        self.update()

    def set_selected_index(self, index):
        index = min(max(index, -1), len(self.annotations) - 1)
        self.selected_index = index
        self.selected_indices = {index} if index >= 0 else set()
        self.selection_changed.emit(index)
        self.update()

    def set_selected_indices(self, indices):
        valid = {index for index in indices if 0 <= index < len(self.annotations)}
        self.selected_indices = valid
        self.selected_index = max(valid) if valid else -1
        self.selection_changed.emit(self.selected_index)
        self.update()

    def select_all_annotations(self):
        if not self.annotations:
            self.set_selected_index(-1)
            return
        self.set_selected_indices(range(len(self.annotations)))

    def toggle_selected_index(self, index):
        if not (0 <= index < len(self.annotations)):
            return
        if index in self.selected_indices:
            self.selected_indices.remove(index)
        else:
            self.selected_indices.add(index)
        self.selected_index = max(self.selected_indices) if self.selected_indices else -1
        self.selection_changed.emit(self.selected_index)
        self.update()

    def selected_annotation(self):
        if 0 <= self.selected_index < len(self.annotations):
            return self.annotations[self.selected_index]
        return None

    def class_name(self, class_id):
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"class_{class_id}"

    def annotation_color(self, class_id):
        hue = (class_id * 47) % 360
        return QColor.fromHsv(hue, 230, 255)

    def annotation_at(self, x, y):
        for index in reversed(range(len(self.annotations))):
            if self.annotations[index].contains(x, y):
                return index
        return -1

    def suggestion_at(self, x, y):
        for index in reversed(range(len(self.suggestions))):
            if self.suggestions[index].contains(x, y):
                return index
        return -1

    def promote_suggestion_to_annotation(self, index):
        if not (0 <= index < len(self.suggestions)):
            return None
        suggestion = self.suggestions.pop(index)
        self.annotations.append(suggestion)
        self.selected_suggestion_index = min(index, len(self.suggestions) - 1)
        self.set_selected_index(len(self.annotations) - 1)
        self.suggestions_changed.emit()
        self.annotations_changed.emit()
        return self.annotations[-1]

    def delete_annotations_in_roi(self):
        if not self.has_roi() or not self.annotations:
            return 0
        self.push_history()
        kept = [annotation for annotation in self.annotations if not self.inside_roi(annotation, self.roi_points)]
        removed = len(self.annotations) - len(kept)
        self.annotations = kept
        self.selected_indices = set()
        self.set_selected_index(min(self.selected_index, len(self.annotations) - 1))
        self.annotations_changed.emit()
        self.update()
        return removed

    def handle_hit(self, annotation, x, y):
        box = annotation.normalized()
        radius = HANDLE_SIZE / max(self.current_scale(), 0.001)
        handles = {
            "tl": (box.x1, box.y1),
            "tr": (box.x2, box.y1),
            "bl": (box.x1, box.y2),
            "br": (box.x2, box.y2),
            "tm": ((box.x1 + box.x2) / 2.0, box.y1),
            "bm": ((box.x1 + box.x2) / 2.0, box.y2),
            "ml": (box.x1, (box.y1 + box.y2) / 2.0),
            "mr": (box.x2, (box.y1 + box.y2) / 2.0),
        }
        for name, point in handles.items():
            if abs(point[0] - x) <= radius and abs(point[1] - y) <= radius:
                return name
        return None

    def set_cursor_for_position(self, image_x, image_y):
        if self.space_pressed:
            self.setCursor(Qt.OpenHandCursor)
            return
        if self.force_new_box_mode and self.has_image():
            self.setCursor(Qt.CrossCursor)
            return
        annotation = self.selected_annotation()
        if annotation:
            handle = self.handle_hit(annotation, image_x, image_y)
            if handle in ("tl", "br"):
                self.setCursor(Qt.SizeFDiagCursor)
                return
            if handle in ("tr", "bl"):
                self.setCursor(Qt.SizeBDiagCursor)
                return
            if handle in ("tm", "bm"):
                self.setCursor(Qt.SizeVerCursor)
                return
            if handle in ("ml", "mr"):
                self.setCursor(Qt.SizeHorCursor)
                return
        hovered = self.annotation_at(image_x, image_y)
        if hovered >= 0:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor if self.has_image() else Qt.ArrowCursor)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 35, 42))

        if not self.has_image():
            painter.setPen(QColor(210, 210, 210))
            painter.setFont(QFont("Segoe UI", max(12, self.ui_font_size + 2)))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open an image folder to start annotating")
            return

        self.update_view_metrics()
        painter.drawPixmap(
            int(self.base_offset_x),
            int(self.base_offset_y),
            int(self.pixmap.width() * self.current_scale()),
            int(self.pixmap.height() * self.current_scale()),
            self.pixmap,
        )

        if self.show_boxes:
            for index, annotation in enumerate(self.annotations):
                self.draw_annotation(painter, annotation, index in self.selected_indices)

        if self.temp_box and self.show_boxes:
            self.draw_annotation(painter, self.temp_box, False, preview=True)

        if self.show_boxes:
            for index, suggestion in enumerate(self.suggestions):
                self.draw_suggestion(painter, suggestion, index == self.selected_suggestion_index)

        if self.has_roi() or (self.roi_mode and self.roi_points):
            self.draw_roi(painter)

    def draw_annotation(self, painter, annotation, is_selected, preview=False):
        box = annotation.normalized()
        dx1, dy1 = self.image_to_display(box.x1, box.y1)
        dx2, dy2 = self.image_to_display(box.x2, box.y2)
        rect = QRectF(dx1, dy1, dx2 - dx1, dy2 - dy1)

        color = QColor(255, 214, 10) if preview else self.annotation_color(annotation.class_id)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(color, 3 if is_selected else 2))
        painter.drawRect(rect)

        label = f"{annotation.class_id}: {self.class_name(annotation.class_id)}"
        label_font_size = max(8, self.ui_font_size - 1)
        painter.setFont(QFont("Segoe UI", label_font_size))
        text_height = max(20, label_font_size + 10)
        text_rect = QRectF(rect.left(), max(0, rect.top() - text_height - 2), max(90, len(label) * max(7, label_font_size - 1)), text_height)
        painter.fillRect(text_rect, QColor(color.red(), color.green(), color.blue(), 180))
        painter.setPen(Qt.black)
        painter.drawText(text_rect.adjusted(6, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, label)

        if is_selected and not preview:
            painter.setPen(QPen(Qt.white, 1))
            painter.setBrush(Qt.white)
            for handle_x, handle_y in self.display_handles(box):
                painter.drawRect(QRectF(handle_x - HANDLE_SIZE / 2, handle_y - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE))

    def draw_suggestion(self, painter, annotation, is_selected):
        box = annotation.normalized()
        dx1, dy1 = self.image_to_display(box.x1, box.y1)
        dx2, dy2 = self.image_to_display(box.x2, box.y2)
        rect = QRectF(dx1, dy1, dx2 - dx1, dy2 - dy1)
        pen = QPen(QColor(0, 220, 255), 3 if is_selected else 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        label_font_size = max(8, self.ui_font_size - 1)
        painter.setFont(QFont("Segoe UI", label_font_size))
        text_height = max(20, label_font_size + 10)
        text_rect = QRectF(rect.left(), max(0, rect.top() - text_height - 2), 220, text_height)
        painter.fillRect(text_rect, QColor(0, 220, 255, 180))
        painter.setPen(Qt.black)
        score_text = f" {annotation.score:.2f}" if annotation.score >= 0 else ""
        painter.drawText(
            text_rect.adjusted(6, 0, -4, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"Suggest: {self.class_name(annotation.class_id)}{score_text}",
        )

    def draw_roi(self, painter):
        polygon_points = [QPointF(*self.image_to_display(x, y)) for x, y in self.roi_points]
        if self.roi_mode and self.roi_preview_point:
            polygon_points.append(QPointF(*self.image_to_display(self.roi_preview_point[0], self.roi_preview_point[1])))
        if not polygon_points:
            return
        painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.DotLine))
        painter.setBrush(QColor(255, 140, 0, 40) if self.has_roi() else Qt.NoBrush)
        if len(polygon_points) >= 3 and self.has_roi():
            painter.drawPolygon(QPolygonF(polygon_points))
        else:
            painter.drawPolyline(QPolygonF(polygon_points))

    def display_handles(self, box):
        if isinstance(box, Annotation):
            box = box.normalized()
        points = [
            (box.x1, box.y1),
            (box.x2, box.y1),
            (box.x1, box.y2),
            (box.x2, box.y2),
            ((box.x1 + box.x2) / 2.0, box.y1),
            ((box.x1 + box.x2) / 2.0, box.y2),
            (box.x1, (box.y1 + box.y2) / 2.0),
            (box.x2, (box.y1 + box.y2) / 2.0),
        ]
        return [self.image_to_display(x, y) for x, y in points]

    def resize_annotation(self, annotation, handle_name, point_x, point_y, symmetric=False, base_box=None):
        box = annotation.normalized()
        base = base_box.normalized() if isinstance(base_box, Annotation) else box
        if "l" in handle_name:
            box.x1 = point_x
        if "r" in handle_name:
            box.x2 = point_x
        if "t" in handle_name:
            box.y1 = point_y
        if "b" in handle_name:
            box.y2 = point_y
        if handle_name == "tm":
            box.y1 = point_y
        if handle_name == "bm":
            box.y2 = point_y
        if handle_name == "ml":
            box.x1 = point_x
        if handle_name == "mr":
            box.x2 = point_x
        if symmetric:
            if handle_name in ("l", "ml"):
                dx = box.x1 - base.x1
                box.x2 = base.x2 - dx
            elif handle_name in ("r", "mr"):
                dx = box.x2 - base.x2
                box.x1 = base.x1 - dx
            elif handle_name in ("t", "tm"):
                dy = box.y1 - base.y1
                box.y2 = base.y2 - dy
            elif handle_name in ("b", "bm"):
                dy = box.y2 - base.y2
                box.y1 = base.y1 - dy
            elif handle_name == "tl":
                dx = box.x1 - base.x1
                dy = box.y1 - base.y1
                box.x2 = base.x2 - dx
                box.y2 = base.y2 - dy
            elif handle_name == "tr":
                dx = box.x2 - base.x2
                dy = box.y1 - base.y1
                box.x1 = base.x1 - dx
                box.y2 = base.y2 - dy
            elif handle_name == "bl":
                dx = box.x1 - base.x1
                dy = box.y2 - base.y2
                box.x2 = base.x2 - dx
                box.y1 = base.y1 - dy
            elif handle_name == "br":
                dx = box.x2 - base.x2
                dy = box.y2 - base.y2
                box.x1 = base.x1 - dx
                box.y1 = base.y1 - dy
        box = box.normalized()
        if box.width() >= MIN_BOX_SIZE and box.height() >= MIN_BOX_SIZE:
            annotation.x1, annotation.y1, annotation.x2, annotation.y2 = box.x1, box.y1, box.x2, box.y2

    def mousePressEvent(self, event):
        if not self.has_image():
            return
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.space_pressed):
            self.mode = "pan"
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        image_x, image_y = self.display_to_image(event.x(), event.y())
        image_x, image_y = self.clamp_point(image_x, image_y)

        if self.roi_mode:
            if event.button() == Qt.RightButton or event.modifiers() & Qt.ControlModifier:
                self.clear_roi()
                return
            if event.button() != Qt.LeftButton:
                return
            self.mode = "roi"
            self.roi_points = [(image_x, image_y)]
            self.roi_preview_point = (image_x, image_y)
            self.roi_changed.emit("ROI lasso: drag to draw free region")
            self.update()
            return

        if event.button() == Qt.RightButton:
            suggestion_index = self.suggestion_at(image_x, image_y)
            if suggestion_index >= 0:
                self.selected_suggestion_index = suggestion_index
                self.reject_selected_suggestion()
            return
        if event.button() != Qt.LeftButton:
            return

        suggestion_index = self.suggestion_at(image_x, image_y)
        if suggestion_index >= 0:
            self.selected_suggestion_index = suggestion_index
            self.set_selected_index(-1)
            self.mode = "suggest_move_pending"
            self.drag_start_image = (image_x, image_y)
            selected = self.suggestions[suggestion_index]
            self.drag_start_box = Annotation(selected.x1, selected.y1, selected.x2, selected.y2, selected.class_id, selected.score)
            self.suggestions_changed.emit()
            self.update()
            return

        annotation = self.selected_annotation()
        if annotation:
            handle = self.handle_hit(annotation, image_x, image_y)
            if handle:
                self.mode = "resize"
                self.active_handle = handle
                self.drag_start_box = Annotation(annotation.x1, annotation.y1, annotation.x2, annotation.y2, annotation.class_id, annotation.score)
                self.push_history()
                return

        hit_index = -1 if self.force_new_box_mode else self.annotation_at(image_x, image_y)
        if hit_index >= 0:
            if event.modifiers() & Qt.ControlModifier:
                if hit_index in self.selected_indices:
                    self.mode = "move_group"
                    self.drag_start_image = (image_x, image_y)
                    self.drag_start_boxes = {
                        index: Annotation(
                            self.annotations[index].x1,
                            self.annotations[index].y1,
                            self.annotations[index].x2,
                            self.annotations[index].y2,
                            self.annotations[index].class_id,
                            self.annotations[index].score,
                        )
                        for index in self.selected_indices
                        if 0 <= index < len(self.annotations)
                    }
                    if self.drag_start_boxes:
                        self.setCursor(Qt.ClosedHandCursor)
                        self.push_history()
                        return
                self.toggle_selected_index(hit_index)
                return
            self.set_selected_index(hit_index)
            self.mode = "move"
            self.drag_start_image = (image_x, image_y)
            selected = self.selected_annotation()
            self.drag_start_box = Annotation(selected.x1, selected.y1, selected.x2, selected.y2, selected.class_id, selected.score)
            self.setCursor(Qt.ClosedHandCursor)
            self.push_history()
            return

        if not (event.modifiers() & Qt.ControlModifier):
            self.set_selected_index(-1)
        self.mode = "draw"
        self.temp_box = Annotation(image_x, image_y, image_x, image_y, self.current_class_id, -1.0)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.has_image():
            return
        image_x, image_y = self.display_to_image(event.x(), event.y())
        image_x, image_y = self.clamp_point(image_x, image_y)
        self.mouse_position_changed.emit(f"x={image_x:.1f}, y={image_y:.1f}, zoom={self.zoom:.2f}x")

        if self.mode == "pan":
            delta = event.pos() - self.last_mouse_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_mouse_pos = event.pos()
            self.update()
            return

        if self.mode == "draw" and self.temp_box:
            self.temp_box.x2 = image_x
            self.temp_box.y2 = image_y
            self.update()
            return

        if self.mode == "roi":
            if not self.roi_points:
                self.roi_points.append((image_x, image_y))
            last_x, last_y = self.roi_points[-1]
            if abs(image_x - last_x) + abs(image_y - last_y) >= 4:
                self.roi_points.append((image_x, image_y))
            self.roi_preview_point = (image_x, image_y)
            self.update()
            return

        if self.mode == "suggest_move_pending" and self.drag_start_image and self.drag_start_box:
            dx = image_x - self.drag_start_image[0]
            dy = image_y - self.drag_start_image[1]
            if abs(dx) > 1 or abs(dy) > 1:
                self.push_history()
                promoted = self.promote_suggestion_to_annotation(self.selected_suggestion_index)
                if promoted is None:
                    return
                self.drag_start_box = Annotation(promoted.x1, promoted.y1, promoted.x2, promoted.y2, promoted.class_id, promoted.score)
                self.mode = "move"
                current = self.selected_annotation()
                if current:
                    self.drag_start_image = (image_x, image_y)
                    self.drag_start_box = Annotation(current.x1, current.y1, current.x2, current.y2, current.class_id, current.score)
            return

        if self.mode == "move_group" and self.drag_start_image and self.drag_start_boxes:
            dx = image_x - self.drag_start_image[0]
            dy = image_y - self.drag_start_image[1]
            h, w = self.image.shape[:2]
            for index, start_box in self.drag_start_boxes.items():
                if (
                    start_box.x1 + dx < 0
                    or start_box.y1 + dy < 0
                    or start_box.x2 + dx > w
                    or start_box.y2 + dy > h
                ):
                    return
            for index, start_box in self.drag_start_boxes.items():
                if 0 <= index < len(self.annotations):
                    self.annotations[index].x1 = start_box.x1 + dx
                    self.annotations[index].y1 = start_box.y1 + dy
                    self.annotations[index].x2 = start_box.x2 + dx
                    self.annotations[index].y2 = start_box.y2 + dy
            self.annotations_changed.emit()
            self.update()
            return

        if self.mode == "move" and self.selected_annotation() and self.drag_start_image and self.drag_start_box:
            dx = image_x - self.drag_start_image[0]
            dy = image_y - self.drag_start_image[1]
            moved = Annotation(
                self.drag_start_box.x1 + dx,
                self.drag_start_box.y1 + dy,
                self.drag_start_box.x2 + dx,
                self.drag_start_box.y2 + dy,
                self.drag_start_box.class_id,
                self.drag_start_box.score,
            )
            moved.clamp(self.image.shape[1], self.image.shape[0])
            current = self.selected_annotation()
            current.x1, current.y1, current.x2, current.y2 = moved.x1, moved.y1, moved.x2, moved.y2
            self.annotations_changed.emit()
            self.update()
            return

        if self.mode == "resize" and self.selected_annotation():
            annotation = self.selected_annotation()
            symmetric = bool(event.modifiers() & Qt.ShiftModifier)
            self.resize_annotation(annotation, self.active_handle, image_x, image_y, symmetric, self.drag_start_box)
            annotation.clamp(self.image.shape[1], self.image.shape[0])
            self.annotations_changed.emit()
            self.update()
            return

        self.set_cursor_for_position(image_x, image_y)

    def mouseReleaseEvent(self, event):
        if not self.has_image():
            return
        if self.mode == "draw" and self.temp_box:
            box = self.temp_box.normalized()
            if box.width() >= MIN_BOX_SIZE and box.height() >= MIN_BOX_SIZE:
                self.push_history()
                self.annotations.append(box)
                self.set_selected_index(len(self.annotations) - 1)
                self.annotations_changed.emit()
        if self.mode == "roi":
            if len(self.roi_points) >= 3:
                self.roi_preview_point = None
                self.set_selected_indices(self.collect_annotations_in_roi())
                self.roi_changed.emit(self.roi_bounds_text())
            else:
                self.clear_roi()
            self.roi_mode = False
        if self.mode in ("move", "resize", "suggest_move_pending"):
            self.annotations_changed.emit()
        self.temp_box = None
        self.mode = "idle"
        self.active_handle = None
        self.drag_start_image = None
        self.drag_start_box = None
        self.drag_start_boxes = None
        image_x, image_y = self.display_to_image(event.x(), event.y())
        image_x, image_y = self.clamp_point(image_x, image_y)
        self.set_cursor_for_position(image_x, image_y)
        self.update()

    def wheelEvent(self, event):
        if not self.has_image():
            return
        before_x, before_y = self.display_to_image(event.x(), event.y())
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.1, min(self.zoom * factor, 25.0))
        self.update_view_metrics()
        after_dx, after_dy = self.image_to_display(before_x, before_y)
        self.pan_x += event.x() - after_dx
        self.pan_y += event.y() - after_dy
        self.update()

    def leaveEvent(self, _event):
        self.mouse_position_changed.emit("")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_B:
            self.force_new_box_mode = True
            self.setCursor(Qt.CrossCursor)
            return
        if event.key() == Qt.Key_Space:
            self.space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
            return
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self.space_pressed:
                step = 20 if (event.modifiers() & Qt.ShiftModifier) else 12
                dx = -step if event.key() == Qt.Key_Left else step if event.key() == Qt.Key_Right else 0
                dy = -step if event.key() == Qt.Key_Up else step if event.key() == Qt.Key_Down else 0
                self.pan_view(dx, dy)
                return
            step = 10 if (event.modifiers() & Qt.ShiftModifier) else 1
            if self.selected_indices or self.selected_index >= 0:
                dx = -step if event.key() == Qt.Key_Left else step if event.key() == Qt.Key_Right else 0
                dy = -step if event.key() == Qt.Key_Up else step if event.key() == Qt.Key_Down else 0
                self.nudge_selected_group(dx, dy)
                return
            if event.key() == Qt.Key_Left:
                self.navigate_requested.emit(-1)
                return
            if event.key() == Qt.Key_Right:
                self.navigate_requested.emit(1)
                return
        if event.key() == Qt.Key_Escape:
            self.mode = "idle"
            self.temp_box = None
            if self.roi_mode:
                self.clear_roi()
            self.update()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.roi_mode and len(self.roi_points) >= 3:
            self.roi_preview_point = None
            self.roi_changed.emit(self.roi_bounds_text())
            self.roi_mode = False
            self.mode = "idle"
            self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_B:
            self.force_new_box_mode = False
            if self.has_image():
                pos = self.mapFromGlobal(QCursor.pos())
                image_x, image_y = self.display_to_image(pos.x(), pos.y())
                image_x, image_y = self.clamp_point(image_x, image_y)
                self.set_cursor_for_position(image_x, image_y)
            else:
                self.unsetCursor()
            return
        if event.key() == Qt.Key_Space:
            self.space_pressed = False
            self.unsetCursor()
            return
        super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.has_image():
            return
        image_x, image_y = self.display_to_image(event.x(), event.y())
        image_x, image_y = self.clamp_point(image_x, image_y)
        suggestion_index = self.suggestion_at(image_x, image_y)
        if suggestion_index >= 0:
            self.selected_suggestion_index = suggestion_index
            self.accept_selected_suggestion()
            return
        hit_index = self.annotation_at(image_x, image_y)
        if hit_index >= 0:
            self.set_selected_index(hit_index)
            class_id = self.annotations[hit_index].class_id
            self.class_edit_requested.emit(class_id)
            return
        super().mouseDoubleClickEvent(event)

    def delete_selected(self):
        if 0 <= self.selected_suggestion_index < len(self.suggestions) and self.selected_index < 0:
            self.reject_selected_suggestion()
            return
        if self.selected_indices:
            self.push_history()
            self.annotations = [annotation for index, annotation in enumerate(self.annotations) if index not in self.selected_indices]
            self.set_selected_index(min(self.selected_index, len(self.annotations) - 1))
            self.annotations_changed.emit()
            self.update()

    def copy_selected(self):
        indices = sorted(self.selected_indices) if self.selected_indices else ([self.selected_index] if self.selected_index >= 0 else [])
        if not indices:
            return
        items = []
        for index in indices:
            if 0 <= index < len(self.annotations):
                annotation = self.annotations[index]
                items.append(
                    Annotation(annotation.x1, annotation.y1, annotation.x2, annotation.y2, annotation.class_id, annotation.score)
                )
        if not items:
            return
        self.clipboard_annotations = items
        self.clipboard_annotation = items[0]

    def paste_annotation(self):
        if not self.clipboard_annotations or not self.has_image():
            return
        self.push_history()
        new_indices = []
        for offset, item in enumerate(self.clipboard_annotations):
            dx = 0
            dy = 0
            clone = Annotation(
                item.x1 + dx,
                item.y1 + dy,
                item.x2 + dx,
                item.y2 + dy,
                item.class_id,
                item.score,
            )
            clone.clamp(self.image.shape[1], self.image.shape[0])
            self.annotations.append(clone)
            new_indices.append(len(self.annotations) - 1)
        if new_indices:
            self.set_selected_indices(new_indices)
        self.annotations_changed.emit()
        self.update()

    def duplicate_selected(self):
        if not self.selected_indices and self.selected_index < 0:
            return
        self.copy_selected()
        self.paste_annotation()

    def cut_selected(self):
        if not self.selected_indices and self.selected_index < 0:
            return
        self.copy_selected()
        self.delete_selected()

    def nudge_selected(self, dx, dy):
        annotation = self.selected_annotation()
        if not annotation or not self.has_image():
            return
        self.push_history()
        annotation.move_by(dx, dy)
        annotation.clamp(self.image.shape[1], self.image.shape[0])
        self.annotations_changed.emit()
        self.update()

    def nudge_selected_group(self, dx, dy):
        if not self.has_image():
            return
        indices = sorted(self.selected_indices) if self.selected_indices else ([self.selected_index] if self.selected_index >= 0 else [])
        if not indices:
            return
        h, w = self.image.shape[:2]
        for index in indices:
            if 0 <= index < len(self.annotations):
                annotation = self.annotations[index]
                if (
                    annotation.x1 + dx < 0
                    or annotation.y1 + dy < 0
                    or annotation.x2 + dx > w
                    or annotation.y2 + dy > h
                ):
                    return
        self.push_history()
        for index in indices:
            if 0 <= index < len(self.annotations):
                self.annotations[index].move_by(dx, dy)
        self.annotations_changed.emit()
        self.update()

    def pan_view(self, dx, dy):
        if not self.has_image():
            return
        self.pan_x += dx
        self.pan_y += dy
        self.update()

    def zoom_by_factor(self, factor):
        if not self.has_image():
            return
        self.zoom = max(0.1, min(self.zoom * factor, 25.0))
        self.update()

    def zoom_in(self):
        self.zoom_by_factor(1.15)

    def zoom_out(self):
        self.zoom_by_factor(1 / 1.15)

    def copy_previous_annotations(self, previous_annotations):
        if not previous_annotations:
            return
        self.push_history()
        start_index = len(self.annotations)
        copied = [Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score) for item in previous_annotations]
        self.annotations.extend(copied)
        if copied:
            self.set_selected_indices(range(start_index, start_index + len(copied)))
        else:
            self.set_selected_index(-1)
        self.annotations_changed.emit()
        self.update()


class Annotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_title = "YOLO Advanced Annotator Pro (Arnob)"
        self.labels_dirty = False
        self.label_signature_cache = {}
        self.loading_image = False
        self.syncing_annotation_list = False
        self.setWindowTitle(self.base_title)
        self.resize(1500, 880)
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.settings_ready = False

        self.folder = self.settings.value("last_folder", "", type=str)
        self.image_files = []
        self.label_cache = {}
        self.reviewed_cache = {}
        self.label_summary_cache = {}
        self.image_feature_cache = {}
        self.image_size_cache = {}
        self.labeled_count = 0
        self.reviewed_count = 0
        self.last_disk_sync_time = 0.0
        self.previous_annotations_cache = []
        self.pending_next_suggestions = None
        self.pending_next_suggestion_image = ""
        self.pending_next_suggestion_count = 0
        self.class_file_path = ""
        self.project_master_classes = []
        self.current_image_index = -1
        self.model_path = ""
        self.detector = None
        self.model_class_names = []
        self.model_class_map = {}
        self.saved_suggest_class_names = []
        self.replace_pending = False
        self.shortcut_config = self.load_shortcut_config()
        self.shortcuts = []
        self.persist_roi_across_images = False
        self.normalized_roi_points = []
        self.keyboard_review_mode = False
        self.auto_advance_mode = "none"
        self.reference_window = 5
        self.last_image_name = self.settings.value("last_image_name", "", type=str)
        self.last_file_list_scroll = self.settings.value("file_list_scroll", 0, type=int)
        self.last_zoom = self.settings.value("canvas_zoom", 1.0, type=float)
        self.last_pan_x = self.settings.value("canvas_pan_x", 0.0, type=float)
        self.last_pan_y = self.settings.value("canvas_pan_y", 0.0, type=float)
        self.skip_save_on_next_load = False
        self.ui_font_size = self.settings.value("ui_font_size", QApplication.font().pointSize(), type=int)
        self.shortcuts_suspended = False
        self.class_type_buffer = ""
        self.class_type_last_key = ""
        self.class_type_last_time = 0.0
        self.class_type_cycle_index = -1
        self.class_type_cycle_query = ""
        self.class_type_timer = QTimer(self)
        self.class_type_timer.setSingleShot(True)
        self.class_type_timer.timeout.connect(self.clear_class_type_buffer)

        self.canvas = Canvas()
        self.canvas.set_ui_font_size(self.ui_font_size)
        self.canvas.annotations_changed.connect(self.refresh_annotation_list)
        self.canvas.annotations_changed.connect(self.on_annotations_changed)
        self.canvas.selection_changed.connect(self.sync_annotation_selection)
        self.canvas.mouse_position_changed.connect(self.update_mouse_status)
        self.canvas.suggestions_changed.connect(self.refresh_suggestion_info)
        self.canvas.roi_changed.connect(self.update_roi_status)
        self.canvas.class_edit_requested.connect(self.focus_class_for_edit)
        self.canvas.navigate_requested.connect(self.navigate_from_canvas)
        self.canvas.navigate_requested.connect(self.navigate_from_canvas)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentRowChanged.connect(self.load_current_image)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.file_list.setTextElideMode(Qt.ElideRight)
        self.file_list.setUniformItemSizes(True)

        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(self.on_annotation_selection_changed)
        self.annotation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.annotation_list.setTextElideMode(Qt.ElideRight)

        self.suggestion_list = QListWidget()
        self.suggestion_list.currentRowChanged.connect(self.on_suggestion_clicked)
        self.suggestion_list.hide()

        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self.on_class_selected)
        self.class_list.itemSelectionChanged.connect(lambda: self.class_list.setFocus())
        self.class_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.class_list.setTextElideMode(Qt.ElideRight)
        self.class_list_row_by_id = {}
        self.class_list.installEventFilter(self)
        self.class_list.viewport().installEventFilter(self)

        self.class_picker = QComboBox()
        self.class_picker.currentIndexChanged.connect(self.on_class_picker_changed)
        self.class_picker.installEventFilter(self)
        self.class_picker_view = None
        try:
            self.class_picker_view = self.class_picker.view()
            if self.class_picker_view is not None:
                self.class_picker_view.installEventFilter(self)
        except Exception:
            self.class_picker_view = None

        self.class_sort_mode = QComboBox()
        self.class_sort_mode.addItem("Sort: ID", "id")
        self.class_sort_mode.addItem("Sort: A-Z", "az")
        self.class_sort_mode.currentIndexChanged.connect(self.on_class_sort_changed)
        self.suggest_class_filter = QListWidget()
        self.suggest_class_filter.setSelectionMode(QAbstractItemView.MultiSelection)
        self.suggest_class_filter.setMaximumHeight(110)
        self.suggest_class_filter.setMinimumHeight(80)
        self.suggest_class_filter.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.suggest_class_filter.setTextElideMode(Qt.ElideRight)

        self.autosave = QCheckBox("Autosave")
        self.autosave.setChecked(True)
        self.autosave_mode = QComboBox()
        self.autosave_mode.addItem("On image change", "image_change")
        self.autosave_mode.addItem("Timed interval", "timed")

        self.show_boxes = QCheckBox("Show Boxes")
        self.show_boxes.setChecked(True)
        self.show_boxes.toggled.connect(self.toggle_boxes)

        self.image_info = QLabel("No folder opened")
        self.image_info.setWordWrap(True)
        self.dataset_info = QLabel("Dataset: 0 images")
        self.dataset_info.setWordWrap(True)
        self.model_info = QLabel("Model: none")
        self.model_info.setWordWrap(True)
        self.suggestion_info = QLabel("Suggestions: 0")
        self.suggestion_info.setWordWrap(True)
        self.roi_info = QLabel("ROI: full image")
        self.roi_info.setWordWrap(True)
        self.roi_hint = QLabel("")
        self.roi_hint.setWordWrap(True)
        self.shortcut_info = QLabel("Draw: drag | Pan: Space+drag | New box over old: hold B | Undo/Redo: Z/Y | Reset: C")
        self.shortcut_info.setWordWrap(True)
        self.autosave_status = QLabel("Autosave: idle")
        self.autosave_status.setWordWrap(True)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setPrefix("Conf ")

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setPrefix("IoU ")
        self.autosave_minutes = QDoubleSpinBox()
        self.autosave_minutes.setRange(0.1, 30.0)
        self.autosave_minutes.setSingleStep(0.1)
        self.autosave_minutes.setValue(1.0)
        self.autosave_minutes.setSuffix(" min")
        self.autosave_minutes.setToolTip("Autosave debounce interval in minutes")
        self.autosave_minutes.valueChanged.connect(self.update_autosave_status)

        self.replace_labels = QCheckBox("Replace labels on auto")
        self.replace_labels.setChecked(False)
        self.auto_next_mode = QComboBox()
        self.auto_next_mode.addItem("No auto on next image", "none")
        self.auto_next_mode.addItem("Auto suggest whole image", "full")
        self.auto_next_mode.addItem("Auto suggest previous ROI", "roi")
        self.auto_next_mode.currentIndexChanged.connect(self.on_auto_next_mode_changed)
        self.sort_mode = QComboBox()
        self.sort_mode.addItem("Name A-Z", "name_asc")
        self.sort_mode.addItem("Name 1-2-10", "name_natural")
        self.sort_mode.addItem("Name Z-A", "name_desc")
        self.sort_mode.currentIndexChanged.connect(self.on_sort_mode_changed)
        self.group_sort_mode = QComboBox()
        self.group_sort_mode.addItem("Group: None", "none")
        self.group_sort_mode.addItem("Group: Labeled first", "labeled_first")
        self.group_sort_mode.addItem("Group: Unlabeled first", "unlabeled_first")
        self.group_sort_mode.addItem("Group: Reviewed first", "reviewed_first")
        self.group_sort_mode.currentIndexChanged.connect(self.on_group_sort_changed)
        self.persist_roi_check = QCheckBox("Keep ROI across images")
        self.persist_roi_check.toggled.connect(self.set_persist_roi)
        self.auto_next_suggest_check = QCheckBox("Auto suggest prev labels")
        self.auto_next_suggest_check.setChecked(False)
        self.reviewed_check = QCheckBox("Reviewed")
        self.reviewed_check.toggled.connect(self.toggle_reviewed)
        self.roi_mode_btn = QPushButton("ROI Select")
        self.roi_mode_btn.setCheckable(True)
        self.roi_mode_btn.toggled.connect(self.canvas.set_roi_mode)
        self.keyboard_review_check = QCheckBox("Keyboard review mode")
        self.keyboard_review_check.toggled.connect(self.set_keyboard_review_mode)
        self.import_model_classes_check = QCheckBox("Auto-import model classes")
        self.import_model_classes_check.setChecked(True)
        self.import_model_classes_check.setToolTip(
            "When enabled, any selected model classes missing from the project will be added before suggesting."
        )

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self.autosave_current_labels)
        self.settings_timer = QTimer(self)
        self.settings_timer.setSingleShot(True)
        self.settings_timer.timeout.connect(self.save_settings)
        self.autosave.toggled.connect(self.update_autosave_status)
        self.autosave_mode.currentIndexChanged.connect(self.update_autosave_status)
        self.autosave_mode.currentIndexChanged.connect(self.update_autosave_controls_visibility)
        self.autosave.toggled.connect(self.schedule_settings_save)
        self.show_boxes.toggled.connect(self.schedule_settings_save)
        self.replace_labels.toggled.connect(self.schedule_settings_save)
        self.autosave_mode.currentIndexChanged.connect(self.schedule_settings_save)
        self.persist_roi_check.toggled.connect(self.schedule_settings_save)
        self.keyboard_review_check.toggled.connect(self.schedule_settings_save)
        self.auto_next_suggest_check.toggled.connect(self.schedule_settings_save)
        self.import_model_classes_check.toggled.connect(self.schedule_settings_save)
        self.auto_next_mode.currentIndexChanged.connect(self.schedule_settings_save)
        self.sort_mode.currentIndexChanged.connect(self.schedule_settings_save)
        self.group_sort_mode.currentIndexChanged.connect(self.schedule_settings_save)
        self.class_sort_mode.currentIndexChanged.connect(self.schedule_settings_save)
        self.conf_spin.valueChanged.connect(self.schedule_settings_save)
        self.iou_spin.valueChanged.connect(self.schedule_settings_save)
        self.autosave_minutes.valueChanged.connect(self.schedule_settings_save)
        self.suggest_class_filter.itemSelectionChanged.connect(self.schedule_settings_save)

        self.setup_layout()
        self.apply_visual_style()
        self.apply_ui_font_size(self.ui_font_size)
        self.setup_actions()
        self.install_shortcuts()
        self.restore_settings()
        self.refresh_annotation_list()
        if self.folder and os.path.isdir(self.folder):
            self.open_existing_folder(self.folder)
        if self.model_path and YOLO is not None and os.path.exists(self.model_path):
            try:
                self.detector = YOLO(self.model_path)
                self.build_model_class_mapping()
            except Exception as exc:
                self.detector = None
                self.status.showMessage(f"Saved model could not be loaded: {exc}", 5000)
        self.settings_ready = True

    def setup_layout(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(230)
        left_panel.setMaximumWidth(320)
        left_layout.addWidget(QLabel("Images"))
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(self.image_info)
        left_layout.addWidget(self.dataset_info)
        left_layout.addWidget(self.sort_mode)
        left_layout.addWidget(self.group_sort_mode)
        nav_row = QHBoxLayout()
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(self.previous_image)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.next_image)
        next_unlabeled_btn = QPushButton("Next Empty")
        next_unlabeled_btn.clicked.connect(self.next_unlabeled_image)
        prev_btn.setToolTip("Previous image")
        next_btn.setToolTip("Next image")
        next_unlabeled_btn.setToolTip("Jump to next unlabeled image")
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        left_layout.addLayout(nav_row)
        left_layout.addWidget(next_unlabeled_btn)
        left_layout.addWidget(self.reviewed_check)
        left_layout.addWidget(self.keyboard_review_check)

        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(390)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(8, 8, 8, 8)
        add_class_btn = QPushButton("Add")
        add_class_btn.clicked.connect(self.add_class)
        rename_class_btn = QPushButton("Rename")
        rename_class_btn.clicked.connect(self.rename_class)
        delete_class_btn = QPushButton("Delete")
        delete_class_btn.clicked.connect(self.delete_class)
        copy_prev_btn = QPushButton("Copy Prev")
        copy_prev_btn.clicked.connect(self.copy_previous_annotations)
        copy_next_suggest_btn = QPushButton("Next Suggest")
        copy_next_suggest_btn.clicked.connect(self.copy_labels_to_next_suggestions)
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo_action)
        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(self.redo_action)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_current_image_annotations)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_current_labels)
        load_model_btn = QPushButton("Load")
        load_model_btn.clicked.connect(self.load_model_file)
        unload_model_btn = QPushButton("Unload")
        unload_model_btn.clicked.connect(self.unload_model)
        auto_current_btn = QPushButton("Suggest")
        auto_current_btn.clicked.connect(self.auto_annotate_current)
        auto_current_btn.setObjectName("primaryButton")
        auto_current_btn.setMinimumHeight(34)
        accept_selected_btn = QPushButton("Accept")
        accept_selected_btn.clicked.connect(self.accept_selected_suggestion)
        accept_all_btn = QPushButton("Accept All")
        accept_all_btn.clicked.connect(self.accept_all_suggestions)
        reject_all_btn = QPushButton("Reject All")
        reject_all_btn.clicked.connect(self.reject_all_suggestions)
        reject_selected_btn = QPushButton("Reject")
        reject_selected_btn.clicked.connect(self.reject_selected_suggestion)
        class_roi_btn = QPushButton("ROI -> Class")
        class_roi_btn.clicked.connect(self.set_labels_in_roi_to_current_class)
        class_selected_btn = QPushButton("Set Class")
        class_selected_btn.clicked.connect(self.set_selected_labels_to_current_class)
        fit_btn = QPushButton("Fit Image")
        fit_btn.clicked.connect(self.canvas.fit_to_window)
        actual_btn = QPushButton("Actual Size")
        actual_btn.clicked.connect(self.canvas.actual_size)
        self.action_buttons = {
            "copy_prev_labels": copy_prev_btn,
            "copy_to_next_suggest": copy_next_suggest_btn,
            "undo": undo_btn,
            "redo": redo_btn,
            "reset": reset_btn,
            "save": save_btn,
            "load_model": load_model_btn,
            "unload_model": unload_model_btn,
            "auto_current": auto_current_btn,
            "accept_suggestion": accept_selected_btn,
            "accept_all_suggestions": accept_all_btn,
            "reject_suggestion": reject_selected_btn,
            "reject_suggestions": reject_all_btn,
            "set_selected_class": class_selected_btn,
            "set_roi_class": class_roi_btn,
            "fit": fit_btn,
            "actual_size": actual_btn,
            "toggle_roi_mode": self.roi_mode_btn,
        }

        project_tab = QWidget()
        project_layout = QVBoxLayout(project_tab)
        project_group = QGroupBox("Classes")
        project_group_layout = QVBoxLayout(project_group)
        project_group_layout.addWidget(self.class_sort_mode)
        project_group_layout.addWidget(self.class_list)
        project_group_layout.addWidget(self.class_picker)
        class_button_row = QHBoxLayout()
        class_button_row.addWidget(add_class_btn)
        class_button_row.addWidget(rename_class_btn)
        class_button_row.addWidget(delete_class_btn)
        project_group_layout.addLayout(class_button_row)
        project_layout.addWidget(project_group)
        project_layout.addWidget(QLabel("Annotations"))
        project_layout.addWidget(self.annotation_list)
        project_action_row = QHBoxLayout()
        project_action_row.addWidget(copy_prev_btn)
        project_action_row.addWidget(copy_next_suggest_btn)
        project_action_row.addWidget(save_btn)
        project_layout.addLayout(project_action_row)
        project_layout.addWidget(self.auto_next_suggest_check)

        workbench_tab = QWidget()
        workbench_layout = QVBoxLayout(workbench_tab)
        review_group = QGroupBox("Review")
        review_group_layout = QVBoxLayout(review_group)
        review_group_layout.setSpacing(6)
        review_group_layout.addWidget(self.suggestion_info)
        review_actions = QVBoxLayout()
        review_actions.setSpacing(6)
        review_row_one = QHBoxLayout()
        review_row_one.addWidget(accept_selected_btn)
        review_row_one.addWidget(reject_selected_btn)
        review_actions.addLayout(review_row_one)
        review_row_two = QHBoxLayout()
        review_row_two.addWidget(accept_all_btn)
        review_row_two.addWidget(reject_all_btn)
        review_actions.addLayout(review_row_two)
        review_row_three = QHBoxLayout()
        review_row_three.addWidget(class_selected_btn)
        review_row_three.addWidget(reset_btn)
        review_actions.addLayout(review_row_three)
        review_group_layout.addLayout(review_actions)
        # Undo/Redo row removed for compact layout
        review_options_row = QHBoxLayout()
        review_options_row.addWidget(self.autosave)
        review_options_row.addWidget(self.show_boxes)
        review_options_row.addStretch(1)
        review_group_layout.addLayout(review_options_row)
        review_group_layout.addWidget(self.autosave_mode)
        review_group_layout.addWidget(self.autosave_minutes)
        review_group_layout.addWidget(self.autosave_status)
        # Shortcut info removed for compact layout
        workbench_layout.addWidget(review_group)
        model_group = QGroupBox("Auto Annotate")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(6)
        model_layout.addWidget(self.model_info)
        auto_row_one = QHBoxLayout()
        auto_row_one.addWidget(load_model_btn)
        auto_row_one.addWidget(unload_model_btn)
        auto_row_one.addWidget(auto_current_btn)
        model_layout.addLayout(auto_row_one)
        auto_row_two = QHBoxLayout()
        auto_row_two.addWidget(self.conf_spin)
        auto_row_two.addWidget(self.iou_spin)
        model_layout.addLayout(auto_row_two)
        replace_row = QHBoxLayout()
        replace_row.addWidget(self.replace_labels)
        replace_row.addStretch(1)
        model_layout.addLayout(replace_row)
        model_layout.addWidget(QLabel("Suggest Classes"))
        model_layout.addWidget(self.suggest_class_filter)
        model_layout.addWidget(self.import_model_classes_check)
        model_layout.addWidget(self.auto_next_mode)
        model_layout.addWidget(self.roi_info)
        model_layout.addWidget(self.roi_hint)
        model_layout.addWidget(self.roi_mode_btn)
        model_layout.addWidget(self.persist_roi_check)
        roi_actions = QVBoxLayout()
        roi_actions.setSpacing(6)
        roi_row_one = QHBoxLayout()
        roi_row_one.addStretch(1)
        roi_actions.addLayout(roi_row_one)
        model_layout.addLayout(roi_actions)
        workbench_layout.addWidget(model_group)
        tools_group = QGroupBox("View")
        tools_layout = QHBoxLayout(tools_group)
        tools_layout.addWidget(fit_btn)
        tools_layout.addWidget(actual_btn)
        workbench_layout.addWidget(tools_group)
        workbench_layout.addStretch(1)

        tabs = QTabWidget()
        tabs.addTab(project_tab, "Project")
        tabs.addTab(workbench_tab, "Review + Auto")
        tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_layout.addWidget(tabs)
        self.tabs = tabs
        self.apply_button_shortcuts()

        self.splitter = QSplitter()
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([250, 1240, 330])
        self.setCentralWidget(self.splitter)

    def setup_actions(self):
        open_folder = QAction("Open Folder", self)
        open_folder.triggered.connect(self.open_folder)

        save_labels = QAction("Save", self)
        save_labels.triggered.connect(self.save_current_labels)

        next_image = QAction("Next Image", self)
        next_image.triggered.connect(self.next_image)

        prev_image = QAction("Previous Image", self)
        prev_image.triggered.connect(self.previous_image)

        load_model = QAction("Load Model", self)
        load_model.triggered.connect(self.load_model_file)
        unload_model = QAction("Unload Model", self)
        unload_model.triggered.connect(self.unload_model)
        export_config = QAction("Export Config XML", self)
        export_config.triggered.connect(self.export_config_xml)
        import_config = QAction("Import Config XML", self)
        import_config.triggered.connect(self.import_config_xml)
        reset_settings = QAction("Reset All Settings", self)
        reset_settings.triggered.connect(self.reset_all_settings)
        configure_shortcuts = QAction("Configure Shortcuts", self)
        configure_shortcuts.triggered.connect(self.configure_shortcuts)
        dataset_integrity = QAction("Dataset Integrity Check", self)
        dataset_integrity.triggered.connect(self.run_dataset_integrity_check)
        fix_out_of_range = QAction("Fix Out-of-Range Labels", self)
        fix_out_of_range.triggered.connect(self.fix_out_of_range_labels)
        increase_font = QAction("Increase Font Size", self)
        increase_font.triggered.connect(lambda: self.change_font_size(1))
        decrease_font = QAction("Decrease Font Size", self)
        decrease_font.triggered.connect(lambda: self.change_font_size(-1))
        reset_font = QAction("Reset Font Size", self)
        reset_font.triggered.connect(self.reset_font_size)
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help_dialog)
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_folder)
        file_menu.addAction(save_labels)
        file_menu.addAction(export_config)
        file_menu.addAction(import_config)
        nav_menu = self.menuBar().addMenu("Navigate")
        nav_menu.addAction(prev_image)
        nav_menu.addAction(next_image)
        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction(load_model)
        tools_menu.addAction(unload_model)
        tools_menu.addAction(dataset_integrity)
        tools_menu.addAction(fix_out_of_range)
        tools_menu.addAction(increase_font)
        tools_menu.addAction(decrease_font)
        tools_menu.addAction(reset_font)
        tools_menu.addAction(reset_settings)
        tools_menu.addAction(configure_shortcuts)
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(help_action)
        help_menu.addAction(about_action)

    def install_shortcuts(self):
        for shortcut in self.shortcuts:
            try:
                shortcut.activated.disconnect()
            except Exception:
                pass
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self.shortcuts = []

        # If duplicate sequences exist, keep the first binding only to avoid
        # stale or conflicting actions after config changes/reset.
        seen_sequences = set()

        bindings = [
            ("open_folder", self.open_folder),
            ("save", self.save_current_labels),
            ("load_model", self.load_model_file),
            ("tab_project", lambda: self.set_active_tab_by_name("Project")),
            ("tab_review", lambda: self.set_active_tab_by_name("Review + Auto")),
            ("select_all_annotations", self.canvas.select_all_annotations),
            ("copy", self.canvas.copy_selected),
            ("cut", self.canvas.cut_selected),
            ("paste", self.canvas.paste_annotation),
            ("duplicate", self.canvas.duplicate_selected),
            ("copy_prev_labels", self.copy_previous_annotations),
            ("copy_to_next_suggest", self.copy_labels_to_next_suggestions),
            ("undo", self.undo_action),
            ("redo", self.redo_action),
            ("reset", self.reset_current_image_annotations),
            ("accept_suggestion", self.accept_selected_suggestion),
            ("reject_suggestion", self.reject_selected_suggestion),
            ("accept_roi_suggestions", self.accept_suggestions_in_roi),
            ("reject_suggestions_only", self.reject_suggestions_only),
            ("delete", self.canvas.delete_selected),
            ("set_selected_class", self.set_selected_labels_to_current_class),
            ("auto_current", self.auto_annotate_current),
            ("clear_roi", self.clear_roi_selection),
            ("toggle_roi_mode", self.toggle_roi_mode_shortcut),
            ("delete_roi", self.delete_labels_in_roi),
            ("set_roi_class", self.set_labels_in_roi_to_current_class),
            ("nudge_left", lambda: self.canvas.nudge_selected_group(-1, 0)),
            ("nudge_right", lambda: self.canvas.nudge_selected_group(1, 0)),
            ("nudge_up", lambda: self.canvas.nudge_selected_group(0, -1)),
            ("nudge_down", lambda: self.canvas.nudge_selected_group(0, 1)),
            ("nudge_left_fast", lambda: self.canvas.nudge_selected_group(-10, 0)),
            ("nudge_right_fast", lambda: self.canvas.nudge_selected_group(10, 0)),
            ("nudge_up_fast", lambda: self.canvas.nudge_selected_group(0, -10)),
            ("nudge_down_fast", lambda: self.canvas.nudge_selected_group(0, 10)),
            ("pan_left", lambda: self.canvas.pan_view(-20, 0)),
            ("pan_right", lambda: self.canvas.pan_view(20, 0)),
            ("pan_up", lambda: self.canvas.pan_view(0, -20)),
            ("pan_down", lambda: self.canvas.pan_view(0, 20)),
            ("zoom_in", self.canvas.zoom_in),
            ("zoom_out", self.canvas.zoom_out),
            ("fit", self.canvas.fit_to_window),
            ("prev_image", self.previous_image),
            ("next_image", self.next_image),
            ("next_unlabeled", self.next_unlabeled_image),
            ("next_unlabeled_alt", self.next_unlabeled_image),
            ("prev_image_alt", self.previous_image),
            ("next_image_alt", self.next_image),
            ("toggle_reviewed", self.toggle_reviewed_shortcut),
            ("accept_all_suggestions", self.accept_all_suggestions),
            ("reject_suggestions", self.reject_all_suggestions),
            ("next_suggestion", lambda: self.canvas.cycle_suggestion(1)),
            ("prev_suggestion", lambda: self.canvas.cycle_suggestion(-1)),
            ("actual_size", self.canvas.actual_size),
        ]
        for key, callback in bindings:
            sequence = self.shortcut_config.get(key, "")
            if not sequence:
                continue
            normalized_sequence = QKeySequence(sequence).toString(QKeySequence.NativeText)
            if not normalized_sequence or normalized_sequence in seen_sequences:
                continue
            seen_sequences.add(normalized_sequence)
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self.guard_shortcut(callback))
            self.shortcuts.append(shortcut)
        for number in range(1, 10):
            sequence = QKeySequence(str(number)).toString(QKeySequence.NativeText)
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            shortcut = QShortcut(QKeySequence(str(number)), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self.guard_shortcut(partial(self.select_class_by_index, number - 1)))
            self.shortcuts.append(shortcut)
        self.apply_button_shortcuts()

    def apply_button_shortcuts(self):
        if not hasattr(self, "action_buttons"):
            return
        button_labels = {
            "copy_prev_labels": "Copy Prev",
            "copy_to_next_suggest": "Next Suggest",
            "undo": "Undo",
            "redo": "Redo",
            "reset": "Reset",
            "save": "Save",
            "load_model": "Load",
            "unload_model": "Unload",
            "auto_current": "Suggest",
            "accept_suggestion": "Accept",
            "accept_all_suggestions": "Accept All",
            "accept_roi_suggestions": "Accept ROI",
            "reject_suggestion": "Reject",
            "reject_suggestions": "Reject All",
            "set_selected_class": "Set Class",
            "clear_roi": "Clear ROI",
            "delete_roi": "Delete ROI",
            "set_roi_class": "ROI -> Class",
            "fit": "Fit Image",
            "actual_size": "Actual Size",
            "toggle_roi_mode": "ROI Select",
        }
        for key, button in self.action_buttons.items():
            sequence = self.shortcut_config.get(key, "")
            button.setText(button_labels[key])
            full_labels = {
                "copy_prev_labels": "Copy labels from previous image",
                "copy_to_next_suggest": "Copy current labels to next image as suggestions",
                "reset": "Reset labels on the current image",
                "save": "Save current labels",
                "load_model": "Load detection model (.pt)",
                "unload_model": "Unload the active detection model",
                "auto_current": "Generate suggestions for current image",
                "accept_suggestion": "Accept selected suggestion",
                "accept_all_suggestions": "Accept all suggestions",
                "accept_roi_suggestions": "Accept suggestions inside ROI",
                "reject_suggestion": "Reject selected suggestion",
                "reject_suggestions": "Reject all suggestions",
                "set_selected_class": "Set selected labels to current class",
                "delete_roi": "Delete labels inside ROI",
                "set_roi_class": "Set ROI labels to current class",
                "toggle_roi_mode": "Toggle ROI select mode",
            }
            tooltip = full_labels.get(key, button_labels[key])
            if sequence:
                tooltip += f"\nShortcut: {sequence}"
            button.setToolTip(tooltip)

    def set_active_tab_by_name(self, name):
        if not hasattr(self, "tabs"):
            return
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == name:
                self.tabs.setCurrentIndex(index)
                break

    def set_shortcuts_enabled(self, enabled):
        for shortcut in getattr(self, "shortcuts", []):
            shortcut.setEnabled(enabled)

    def guard_shortcut(self, callback):
        def wrapped():
            focus_widget = QApplication.focusWidget()
            if self.shortcuts_suspended or self.is_class_focus(focus_widget):
                return
            callback()
        return wrapped

    def is_class_focus(self, widget):
        current = widget
        while current is not None:
            if current in (self.class_list, self.class_picker):
                return True
            current = current.parent()
        return False

    def apply_visual_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f4f6;
            }
            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #d7dce2;
                padding: 4px 8px;
                spacing: 8px;
                color: #1f2937;
                font-weight: 600;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QMenuBar::item:selected {
                background: #e6eef8;
            }
            QMenu {
                background: #ffffff;
                border: 1px solid #d7dce2;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 18px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #e6eef8;
            }
            QGroupBox {
                border: 1px solid #d7dce2;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: #fbfbfc;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #374151;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                padding: 6px 10px;
                color: #1f2937;
            }
            QPushButton:hover {
                background: #f8fafc;
                border-color: #94a3b8;
            }
            QPushButton:checked {
                background: #dbeafe;
                border-color: #2563eb;
                color: #1d4ed8;
                font-weight: 700;
            }
            QPushButton#primaryButton {
                background: #198754;
                border: 1px solid #157347;
                color: white;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: #157347;
                border-color: #146c43;
            }
            QTabWidget::pane {
                border: 1px solid #d7dce2;
                background: #f8fafc;
            }
            QTabBar::tab {
                background: #eef2f7;
                border: 1px solid #d7dce2;
                padding: 6px 12px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
            }
            """
        )

    def update_autosave_status(self, checked=None):
        enabled = self.autosave.isChecked() if checked is None else checked
        if not enabled:
            self.autosave_status.setText("Autosave: off")
            return
        if not self.labels_dirty:
            self.autosave_status.setText("Autosave: up to date")
            return
        mode = self.autosave_mode.currentData() or "image_change"
        if mode == "image_change":
            self.autosave_status.setText("Autosave: on image change")
            return
        if self.autosave_timer.isActive():
            self.autosave_status.setText(f"Autosave: pending ({self.autosave_minutes.value():.1f} min)")
        else:
            self.autosave_status.setText(f"Autosave: ready ({self.autosave_minutes.value():.1f} min)")

    def update_window_title(self):
        suffix = " *" if self.labels_dirty else ""
        self.setWindowTitle(f"{self.base_title}{suffix}")

    def compute_label_signature(self, annotations):
        if not annotations:
            return ()
        return tuple(
            (
                int(annotation.class_id),
                round(annotation.x1, 2),
                round(annotation.y1, 2),
                round(annotation.x2, 2),
                round(annotation.y2, 2),
            )
            for annotation in annotations
        )

    def set_current_labels_clean(self, image_name=None, update_status=True):
        image_name = image_name or self.current_image_name()
        if not image_name:
            return
        signature = self.compute_label_signature(self.canvas.annotations)
        self.label_signature_cache[image_name] = signature
        self.labels_dirty = False
        self.update_window_title()
        if update_status:
            self.update_autosave_status()

    def on_annotations_changed(self):
        if self.loading_image:
            return
        image_name = self.current_image_name()
        if not image_name:
            if self.labels_dirty:
                self.labels_dirty = False
                self.update_window_title()
            self.update_autosave_status()
            return
        signature = self.compute_label_signature(self.canvas.annotations)
        dirty = signature != self.label_signature_cache.get(image_name)
        if dirty != self.labels_dirty:
            self.labels_dirty = dirty
            self.update_window_title()
        self.update_autosave_status()

    def mark_autosave_saved(self, source="autosave"):
        time_text = datetime.now().strftime("%H:%M:%S")
        prefix = "Autosave" if source == "autosave" else "Saved"
        self.autosave_status.setText(f"{prefix}: {time_text}")

    def autosave_on_image_change(self):
        return self.autosave.isChecked() and (self.autosave_mode.currentData() or "image_change") == "image_change"

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", self.folder or os.getcwd())
        if not folder:
            return
        self.open_existing_folder(folder)

    def open_existing_folder(self, folder):
        self.folder = folder
        self.load_classes()
        self.image_files = [entry.name for entry in os.scandir(folder) if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS)]
        self.refresh_dataset_cache()
        self.apply_sort_mode()
        self.refresh_file_list()
        self.image_info.setText(f"{len(self.image_files)} images in {folder}")
        if self.image_files:
            target_name = self.last_image_name if self.last_image_name in self.image_files else self.image_files[0]
            self.file_list.setCurrentRow(self.image_files.index(target_name))
            self.file_list.verticalScrollBar().setValue(self.last_file_list_scroll)
            if self.current_image_name() == target_name:
                self.canvas.zoom = self.last_zoom
                self.canvas.pan_x = self.last_pan_x
                self.canvas.pan_y = self.last_pan_y
                self.canvas.update()
        else:
            self.canvas.image = None
            self.canvas.update()
            QMessageBox.information(self, "No images", "No supported image files were found in this folder.")
        self.save_settings()

    def compute_label_state(self, image_name):
        label_path = self.label_path_for_image(image_name)
        if not os.path.exists(label_path):
            return False
        try:
            return os.path.getsize(label_path) > 0
        except OSError:
            return False

    def compute_reviewed_state(self, image_name):
        return os.path.exists(self.reviewed_path_for_image(image_name))

    def refresh_dataset_cache(self):
        self.label_cache = {}
        self.reviewed_cache = {}
        self.label_summary_cache = {}
        self.image_feature_cache = {}
        self.image_size_cache = {}
        for image_name in self.image_files:
            self.label_cache[image_name] = self.compute_label_state(image_name)
            self.reviewed_cache[image_name] = self.compute_reviewed_state(image_name)
        self.labeled_count = sum(1 for value in self.label_cache.values() if value)
        self.reviewed_count = sum(1 for value in self.reviewed_cache.values() if value)

    def apply_sort_mode(self):
        name_mode = self.sort_mode.currentData() or "name_asc"
        group_mode = self.group_sort_mode.currentData() if hasattr(self, "group_sort_mode") else "none"

        if name_mode == "name_desc":
            self.image_files = sorted(self.image_files, key=lambda name: name.lower(), reverse=True)
        elif name_mode == "name_natural":
            self.image_files = sorted(self.image_files, key=self.natural_sort_key)
        else:
            self.image_files = sorted(self.image_files, key=lambda name: name.lower())

        if group_mode == "labeled_first":
            self.image_files = sorted(self.image_files, key=lambda name: not self.has_saved_labels(name))
        elif group_mode == "unlabeled_first":
            self.image_files = sorted(self.image_files, key=lambda name: self.has_saved_labels(name))
        elif group_mode == "reviewed_first":
            self.image_files = sorted(self.image_files, key=lambda name: not self.is_reviewed_image(name))

    def natural_sort_key(self, value):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

    def on_sort_mode_changed(self, _index):
        if not self.image_files:
            return
        current_name = self.current_image_name()
        self.apply_sort_mode()
        self.refresh_file_list(preserve_row=False)
        if current_name and current_name in self.image_files:
            new_index = self.image_files.index(current_name)
            self.file_list.setCurrentRow(new_index)

    def on_group_sort_changed(self, _index):
        if not self.image_files:
            return
        current_name = self.current_image_name()
        self.apply_sort_mode()
        self.refresh_file_list(preserve_row=False)
        if current_name and current_name in self.image_files:
            new_index = self.image_files.index(current_name)
            self.file_list.setCurrentRow(new_index)

    def load_shortcut_config(self):
        config = dict(DEFAULT_SHORTCUTS)
        for key, default_value in DEFAULT_SHORTCUTS.items():
            stored_value = self.settings.value(f"shortcuts/{key}", default_value, type=str)
            legacy_value = LEGACY_DEFAULT_SHORTCUTS.get(key)
            if key == "prev_image_alt" and stored_value == "Left":
                stored_value = default_value
            if key == "next_image_alt" and stored_value == "Right":
                stored_value = default_value
            config[key] = default_value if legacy_value and stored_value == legacy_value else stored_value
        return config

    def save_settings(self):
        if not getattr(self, "settings_ready", False):
            return
        self.settings.setValue("last_folder", self.folder)
        self.settings.setValue("last_image_name", self.current_image_name())
        self.settings.setValue("file_list_scroll", self.file_list.verticalScrollBar().value())
        self.settings.setValue("canvas_zoom", self.canvas.zoom)
        self.settings.setValue("canvas_pan_x", self.canvas.pan_x)
        self.settings.setValue("canvas_pan_y", self.canvas.pan_y)
        self.settings.setValue("geometry", self.saveGeometry())
        if hasattr(self, "splitter"):
            self.settings.setValue("splitter", self.splitter.saveState())
        if hasattr(self, "tabs"):
            self.settings.setValue("active_tab", self.tabs.currentIndex())
        self.settings.setValue("autosave", self.autosave.isChecked())
        self.settings.setValue("show_boxes", self.show_boxes.isChecked())
        self.settings.setValue("replace_labels", self.replace_labels.isChecked())
        self.settings.setValue("autosave_mode", self.autosave_mode.currentData() or "image_change")
        self.settings.setValue("persist_roi", self.persist_roi_check.isChecked())
        self.settings.setValue("auto_next_suggest", self.auto_next_suggest_check.isChecked())
        self.settings.setValue("keyboard_review", self.keyboard_review_check.isChecked())
        self.settings.setValue("import_model_classes", self.import_model_classes_check.isChecked())
        self.settings.setValue("auto_next_mode", self.auto_advance_mode)
        self.settings.setValue("sort_mode", self.sort_mode.currentData() or "name_asc")
        self.settings.setValue("group_sort_mode", self.group_sort_mode.currentData() or "none")
        self.settings.setValue("class_sort_mode", self.class_sort_mode.currentData() or "id")
        self.settings.setValue("conf", self.conf_spin.value())
        self.settings.setValue("iou", self.iou_spin.value())
        self.settings.setValue("autosave_minutes", self.autosave_minutes.value())
        self.settings.setValue("model_path", self.model_path)
        self.settings.setValue("ui_font_size", self.ui_font_size)
        selected_filter_names = [item.text() for item in self.suggest_class_filter.selectedItems()]
        self.settings.setValue("suggest_class_filter", "\n".join(selected_filter_names))
        for key, value in self.shortcut_config.items():
            self.settings.setValue(f"shortcuts/{key}", value)
        self.settings.sync()

    def schedule_settings_save(self, *_args):
        if not getattr(self, "settings_ready", False):
            return
        self.settings_timer.start(300)

    def restore_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.autosave.setChecked(self.settings.value("autosave", True, type=bool))
        self.show_boxes.setChecked(self.settings.value("show_boxes", True, type=bool))
        self.replace_labels.setChecked(self.settings.value("replace_labels", False, type=bool))
        autosave_mode = self.settings.value("autosave_mode", "image_change", type=str)
        self.autosave_mode.setCurrentIndex(max(0, self.autosave_mode.findData(autosave_mode)))
        self.persist_roi_check.setChecked(self.settings.value("persist_roi", False, type=bool))
        self.auto_next_suggest_check.setChecked(self.settings.value("auto_next_suggest", False, type=bool))
        self.keyboard_review_check.setChecked(self.settings.value("keyboard_review", False, type=bool))
        self.import_model_classes_check.setChecked(self.settings.value("import_model_classes", True, type=bool))
        self.auto_advance_mode = self.settings.value("auto_next_mode", "none", type=str)
        combo_index = max(0, self.auto_next_mode.findData(self.auto_advance_mode))
        self.auto_next_mode.setCurrentIndex(combo_index)
        sort_mode = self.settings.value("sort_mode", "name_asc", type=str)
        if sort_mode in {"labeled_first", "unlabeled_first", "reviewed_first"}:
            self.group_sort_mode.setCurrentIndex(max(0, self.group_sort_mode.findData(sort_mode)))
            sort_mode = "name_asc"
        sort_index = max(0, self.sort_mode.findData(sort_mode))
        self.sort_mode.setCurrentIndex(sort_index)
        group_sort_mode = self.settings.value("group_sort_mode", "none", type=str)
        group_index = max(0, self.group_sort_mode.findData(group_sort_mode))
        self.group_sort_mode.setCurrentIndex(group_index)
        class_sort_mode = self.settings.value("class_sort_mode", "id", type=str)
        class_sort_index = max(0, self.class_sort_mode.findData(class_sort_mode))
        self.class_sort_mode.setCurrentIndex(class_sort_index)
        self.conf_spin.setValue(self.settings.value("conf", 0.25, type=float))
        self.iou_spin.setValue(self.settings.value("iou", 0.45, type=float))
        self.autosave_minutes.setValue(self.settings.value("autosave_minutes", 1.0, type=float))
        self.apply_ui_font_size(self.settings.value("ui_font_size", self.ui_font_size, type=int))
        self.model_path = self.settings.value("model_path", "", type=str)
        saved_filter = self.settings.value("suggest_class_filter", "", type=str)
        self.saved_suggest_class_names = [line for line in saved_filter.splitlines() if line.strip()]
        if self.model_path:
            self.model_info.setText(f"Model: {os.path.basename(self.model_path)}")
        splitter_state = self.settings.value("splitter")
        if splitter_state and hasattr(self, "splitter"):
            self.splitter.restoreState(splitter_state)
        active_tab = self.settings.value("active_tab", 0, type=int)
        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(max(0, min(active_tab, self.tabs.count() - 1)))
        self.update_autosave_status()
        self.update_autosave_controls_visibility()

    def configure_shortcuts(self):
        dialog = ShortcutDialog(self.shortcut_config, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        values = dialog.values()
        self.shortcut_config.update(values)
        self.install_shortcuts()
        self.save_settings()
        self.status.showMessage("Shortcuts updated.", 3000)

    def reset_all_settings(self):
        answer = QMessageBox.question(
            self,
            "Reset All Settings",
            "Reset saved app settings and shortcuts back to defaults? Project labels will not be deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        current_folder = self.folder
        self.settings.clear()
        self.shortcut_config = dict(DEFAULT_SHORTCUTS)
        self.install_shortcuts()
        self.autosave.setChecked(True)
        self.show_boxes.setChecked(True)
        self.replace_labels.setChecked(False)
        self.autosave_mode.setCurrentIndex(max(0, self.autosave_mode.findData("image_change")))
        self.autosave_minutes.setValue(1.0)
        self.persist_roi_check.setChecked(False)
        self.auto_next_suggest_check.setChecked(False)
        self.keyboard_review_check.setChecked(False)
        self.import_model_classes_check.setChecked(True)
        self.auto_advance_mode = "none"
        self.auto_next_mode.setCurrentIndex(max(0, self.auto_next_mode.findData("none")))
        self.sort_mode.setCurrentIndex(max(0, self.sort_mode.findData("name_asc")))
        self.group_sort_mode.setCurrentIndex(max(0, self.group_sort_mode.findData("none")))
        self.class_sort_mode.setCurrentIndex(max(0, self.class_sort_mode.findData("id")))
        self.conf_spin.setValue(0.25)
        self.iou_spin.setValue(0.45)
        self.apply_ui_font_size(QApplication.font().pointSize())
        self.model_path = ""
        self.detector = None
        self.model_class_names = []
        self.model_class_map = {}
        self.model_info.setText("Model: none")
        self.suggest_class_filter.clearSelection()
        self.tabs.setCurrentIndex(0)
        self.splitter.setSizes([250, 1240, 330])
        self.last_image_name = ""
        self.last_file_list_scroll = 0
        self.last_zoom = 1.0
        self.last_pan_x = 0.0
        self.last_pan_y = 0.0
        self.update_autosave_status()
        self.save_settings()
        self.status.showMessage("All saved settings were reset to defaults.", 5000)
        if current_folder and os.path.isdir(current_folder):
            self.open_existing_folder(current_folder)

    def export_config_xml(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Config XML",
            os.path.join(self.folder or os.getcwd(), "annotator_config.xml"),
            "XML Files (*.xml)",
        )
        if not path:
            return
        root = ET.Element("annotator_config", version="1")
        settings_node = ET.SubElement(root, "settings")
        values = {
            "last_folder": self.folder,
            "last_image_name": self.current_image_name() or "",
            "file_list_scroll": str(self.file_list.verticalScrollBar().value()),
            "canvas_zoom": str(self.canvas.zoom),
            "canvas_pan_x": str(self.canvas.pan_x),
            "canvas_pan_y": str(self.canvas.pan_y),
            "active_tab": str(self.tabs.currentIndex() if hasattr(self, "tabs") else 0),
            "autosave": str(self.autosave.isChecked()).lower(),
            "show_boxes": str(self.show_boxes.isChecked()).lower(),
            "replace_labels": str(self.replace_labels.isChecked()).lower(),
            "autosave_mode": self.autosave_mode.currentData() or "image_change",
            "autosave_minutes": str(self.autosave_minutes.value()),
            "persist_roi": str(self.persist_roi_check.isChecked()).lower(),
            "auto_next_suggest": str(self.auto_next_suggest_check.isChecked()).lower(),
            "keyboard_review": str(self.keyboard_review_check.isChecked()).lower(),
            "import_model_classes": str(self.import_model_classes_check.isChecked()).lower(),
            "auto_next_mode": self.auto_advance_mode,
            "sort_mode": self.sort_mode.currentData() or "name_asc",
            "group_sort_mode": self.group_sort_mode.currentData() or "none",
            "class_sort_mode": self.class_sort_mode.currentData() or "id",
            "conf": str(self.conf_spin.value()),
            "iou": str(self.iou_spin.value()),
            "import_model_classes": str(self.import_model_classes_check.isChecked()).lower(),
            "model_path": self.model_path,
            "ui_font_size": str(self.ui_font_size),
            "suggest_class_filter": "\n".join(item.text() for item in self.suggest_class_filter.selectedItems()),
            "geometry_b64": encode_state_bytes(self.saveGeometry()),
            "splitter_b64": encode_state_bytes(self.splitter.saveState() if hasattr(self, "splitter") else None),
        }
        for key, value in values.items():
            node = ET.SubElement(settings_node, "value", key=key)
            node.text = value

        shortcuts_node = ET.SubElement(root, "shortcuts")
        for key, value in sorted(self.shortcut_config.items()):
            node = ET.SubElement(shortcuts_node, "shortcut", key=key)
            node.text = value

        if self.project_master_classes:
            classes_node = ET.SubElement(root, "project_classes")
            for index, class_name in enumerate(self.project_master_classes):
                node = ET.SubElement(classes_node, "class", id=str(index))
                node.text = class_name

        tree = ET.ElementTree(root)
        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass
        tree.write(path, encoding="utf-8", xml_declaration=True)
        self.status.showMessage(f"Exported config to {os.path.basename(path)}", 4000)

    def import_config_xml(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Config XML",
            self.folder or os.getcwd(),
            "XML Files (*.xml)",
        )
        if not path:
            return
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", f"Could not read XML config: {exc}")
            return
        if root.tag != "annotator_config":
            QMessageBox.warning(self, "Import failed", "This XML file is not a compatible annotator config.")
            return

        values = {node.get("key"): (node.text or "") for node in root.findall("./settings/value")}
        imported_folder = values.get("last_folder", self.folder)
        imported_image_name = values.get("last_image_name", self.last_image_name)
        imported_filter = values.get("suggest_class_filter", "")
        imported_geometry = decode_state_bytes(values.get("geometry_b64", ""))
        imported_splitter = decode_state_bytes(values.get("splitter_b64", ""))
        self.autosave.setChecked(values.get("autosave", str(self.autosave.isChecked()).lower()) == "true")
        self.show_boxes.setChecked(values.get("show_boxes", str(self.show_boxes.isChecked()).lower()) == "true")
        self.replace_labels.setChecked(values.get("replace_labels", str(self.replace_labels.isChecked()).lower()) == "true")
        autosave_mode = values.get("autosave_mode", self.autosave_mode.currentData() or "image_change")
        self.autosave_mode.setCurrentIndex(max(0, self.autosave_mode.findData(autosave_mode)))
        self.persist_roi_check.setChecked(values.get("persist_roi", str(self.persist_roi_check.isChecked()).lower()) == "true")
        self.auto_next_suggest_check.setChecked(values.get("auto_next_suggest", str(self.auto_next_suggest_check.isChecked()).lower()) == "true")
        self.keyboard_review_check.setChecked(values.get("keyboard_review", str(self.keyboard_review_check.isChecked()).lower()) == "true")
        self.import_model_classes_check.setChecked(values.get("import_model_classes", str(self.import_model_classes_check.isChecked()).lower()) == "true")
        self.auto_advance_mode = values.get("auto_next_mode", self.auto_advance_mode)
        self.auto_next_mode.setCurrentIndex(max(0, self.auto_next_mode.findData(self.auto_advance_mode)))
        sort_mode = values.get("sort_mode", self.sort_mode.currentData() or "name_asc")
        if sort_mode in {"labeled_first", "unlabeled_first", "reviewed_first"}:
            self.group_sort_mode.setCurrentIndex(max(0, self.group_sort_mode.findData(sort_mode)))
            sort_mode = "name_asc"
        self.sort_mode.setCurrentIndex(max(0, self.sort_mode.findData(sort_mode)))
        group_sort_mode = values.get("group_sort_mode", self.group_sort_mode.currentData() or "none")
        self.group_sort_mode.setCurrentIndex(max(0, self.group_sort_mode.findData(group_sort_mode)))
        class_sort_mode = values.get("class_sort_mode", self.class_sort_mode.currentData() or "id")
        self.class_sort_mode.setCurrentIndex(max(0, self.class_sort_mode.findData(class_sort_mode)))
        try:
            self.conf_spin.setValue(float(values.get("conf", self.conf_spin.value())))
        except (TypeError, ValueError):
            pass
        try:
            self.iou_spin.setValue(float(values.get("iou", self.iou_spin.value())))
        except (TypeError, ValueError):
            pass
        try:
            self.autosave_minutes.setValue(float(values.get("autosave_minutes", self.autosave_minutes.value())))
        except (TypeError, ValueError):
            pass
        try:
            self.last_file_list_scroll = int(values.get("file_list_scroll", self.last_file_list_scroll))
        except (TypeError, ValueError):
            pass
        try:
            self.last_zoom = float(values.get("canvas_zoom", self.last_zoom))
        except (TypeError, ValueError):
            pass
        try:
            self.last_pan_x = float(values.get("canvas_pan_x", self.last_pan_x))
        except (TypeError, ValueError):
            pass
        try:
            self.last_pan_y = float(values.get("canvas_pan_y", self.last_pan_y))
        except (TypeError, ValueError):
            pass
        try:
            self.apply_ui_font_size(int(values.get("ui_font_size", self.ui_font_size)))
        except (TypeError, ValueError):
            pass
        imported_model_path = values.get("model_path", "")
        self.model_path = imported_model_path
        self.last_image_name = imported_image_name or self.last_image_name
        self.saved_suggest_class_names = [line for line in imported_filter.splitlines() if line.strip()]

        imported_shortcuts = {node.get("key"): (node.text or "") for node in root.findall("./shortcuts/shortcut")}
        self.shortcut_config.update(imported_shortcuts)
        self.install_shortcuts()

        if imported_folder and os.path.isdir(imported_folder):
            self.folder = imported_folder
            self.open_existing_folder(imported_folder)

        imported_classes = [node.text.strip() for node in root.findall("./project_classes/class") if node.text and node.text.strip()]
        if imported_classes and self.folder:
            self.project_master_classes = list(imported_classes)
            self.save_classes(self.project_master_classes)
            self.backup_project_classes(force=True)
            self.populate_classes(list(self.project_master_classes))

        if not self.model_path:
            self.detector = None
            self.model_class_names = []
            self.model_class_map = {}
            self.model_info.setText("Model: none")
        elif YOLO is not None and os.path.exists(self.model_path):
            try:
                self.detector = YOLO(self.model_path)
                self.build_model_class_mapping()
            except Exception as exc:
                self.detector = None
                QMessageBox.warning(self, "Model restore failed", f"Could not load saved model: {exc}")
        else:
            self.detector = None
            self.model_class_names = []
            self.model_class_map = {}
            self.model_info.setText(f"Model: missing ({os.path.basename(self.model_path)})")

        try:
            active_tab = int(values.get("active_tab", self.tabs.currentIndex() if hasattr(self, "tabs") else 0))
        except (TypeError, ValueError):
            active_tab = self.tabs.currentIndex() if hasattr(self, "tabs") else 0
        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(max(0, min(active_tab, self.tabs.count() - 1)))
        if imported_geometry:
            self.restoreGeometry(imported_geometry)
        if imported_splitter and hasattr(self, "splitter"):
            self.splitter.restoreState(imported_splitter)

        self.save_settings()
        self.status.showMessage(f"Imported config from {os.path.basename(path)}", 4000)

    def run_dataset_integrity_check(self):
        if not self.folder or not self.image_files:
            QMessageBox.information(self, "Dataset Integrity", "Open a project folder first.")
            return
        unreadable_images = []
        malformed_labels = []
        out_of_range_classes = []
        orphan_labels = []
        label_files = {
            entry.name
            for entry in os.scandir(self.folder)
            if entry.is_file() and entry.name.lower().endswith(".txt") and entry.name not in {"classes.txt", "obj.names", "classes.project_backup.txt"}
        }
        valid_label_files = set()
        class_limit = len(self.canvas.classes)
        for index, image_name in enumerate(self.image_files, start=1):
            size = self.get_image_size(image_name)
            if size is None:
                unreadable_images.append(image_name)
                continue
            label_path = self.label_path_for_image(image_name)
            label_name = os.path.basename(label_path)
            if not os.path.exists(label_path):
                continue
            valid_label_files.add(label_name)
            with open(label_path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        malformed_labels.append(f"{label_name}:{line_number}")
                        continue
                    try:
                        class_id, xc, yc, bw, bh = map(float, parts)
                    except ValueError:
                        malformed_labels.append(f"{label_name}:{line_number}")
                        continue
                    if bw <= 0 or bh <= 0 or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        malformed_labels.append(f"{label_name}:{line_number}")
                        continue
                    int_class_id = int(class_id)
                    if int_class_id < 0 or int_class_id >= class_limit:
                        out_of_range_classes.append(f"{label_name}:{line_number} -> {int_class_id}")
            if index % 50 == 0:
                QApplication.processEvents()
        orphan_labels = sorted(label_files - valid_label_files)
        lines = [
            f"Images scanned: {len(self.image_files)}",
            f"Unreadable images: {len(unreadable_images)}",
            f"Malformed label rows: {len(malformed_labels)}",
            f"Out-of-range class IDs: {len(out_of_range_classes)}",
            f"Orphan label files: {len(orphan_labels)}",
            "",
        ]
        if unreadable_images:
            lines.append("Unreadable images:")
            lines.extend(f"- {name}" for name in unreadable_images[:12])
            if len(unreadable_images) > 12:
                lines.append(f"- ... and {len(unreadable_images) - 12} more")
            lines.append("")
        if malformed_labels:
            lines.append("Malformed label rows:")
            lines.extend(f"- {item}" for item in malformed_labels[:12])
            if len(malformed_labels) > 12:
                lines.append(f"- ... and {len(malformed_labels) - 12} more")
            lines.append("")
        if out_of_range_classes:
            lines.append("Out-of-range class IDs:")
            lines.extend(f"- {item}" for item in out_of_range_classes[:12])
            if len(out_of_range_classes) > 12:
                lines.append(f"- ... and {len(out_of_range_classes) - 12} more")
            lines.append("")
        if orphan_labels:
            lines.append("Orphan label files:")
            lines.extend(f"- {item}" for item in orphan_labels[:12])
            if len(orphan_labels) > 12:
                lines.append(f"- ... and {len(orphan_labels) - 12} more")
        if len(lines) == 6:
            lines.append("No integrity problems found.")
        QMessageBox.information(self, "Dataset Integrity", "\n".join(lines))

    def fix_out_of_range_labels(self):
        if not self.folder or not self.image_files:
            QMessageBox.information(self, "Fix Out-of-Range Labels", "Open a project folder first.")
            return
        class_limit = len(self.canvas.classes)
        if class_limit <= 0:
            QMessageBox.information(self, "Fix Out-of-Range Labels", "No classes are available to validate labels.")
            return
        total_out_of_range = 0
        files_with_out_of_range = 0
        for image_name in self.image_files:
            label_path = self.label_path_for_image(image_name)
            if not os.path.exists(label_path):
                continue
            file_has_issue = False
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if class_id < 0 or class_id >= class_limit:
                        total_out_of_range += 1
                        file_has_issue = True
            if file_has_issue:
                files_with_out_of_range += 1

        answer = QMessageBox.question(
            self,
            "Fix Out-of-Range Labels",
            f"Found {total_out_of_range} out-of-range labels in {files_with_out_of_range} files.\nRemove them?",
        )
        if answer != QMessageBox.Yes:
            return

        removed_entries = 0
        fixed_files = 0
        for image_name in self.image_files:
            label_path = self.label_path_for_image(image_name)
            if not os.path.exists(label_path):
                continue
            kept_lines = []
            removed_here = 0
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if class_id < 0 or class_id >= class_limit:
                        removed_here += 1
                        continue
                    kept_lines.append(" ".join(parts) + "\n")
            if removed_here:
                removed_entries += removed_here
                if kept_lines:
                    atomic_write_lines(label_path, kept_lines)
                else:
                    try:
                        os.remove(label_path)
                    except OSError:
                        pass
                fixed_files += 1
                self.invalidate_label_summary(image_name)
                self.update_counts_for_label_state(image_name, len(kept_lines) > 0)
                self.update_file_list_item(image_name)

        if self.canvas.has_image():
            before = len(self.canvas.annotations)
            self.canvas.annotations = [
                ann for ann in self.canvas.annotations if 0 <= ann.class_id < class_limit
            ]
            if len(self.canvas.annotations) != before:
                self.canvas.set_selected_index(-1)
                self.canvas.annotations_changed.emit()
                self.canvas.update()

        self.refresh_summary_labels()
        QMessageBox.information(
            self,
            "Fix Out-of-Range Labels",
            f"Removed {removed_entries} out-of-range entries across {fixed_files} files.",
        )

    def apply_ui_font_size(self, point_size):
        self.ui_font_size = max(8, min(int(point_size), 24))
        app = QApplication.instance()
        font = None
        if app is not None:
            font = QFont(app.font())
            font.setPointSize(self.ui_font_size)
            app.setFont(font)
        if font is None:
            font = QFont(self.font())
            font.setPointSize(self.ui_font_size)
        self.setFont(font)
        for widget in self.findChildren(QWidget):
            if widget is self.canvas:
                continue
            widget.setFont(font)
        menu_bar = self.menuBar()
        if menu_bar is not None:
            menu_bar.setFont(font)
            for action in menu_bar.actions():
                menu = action.menu()
                if menu is not None:
                    menu.setFont(font)
        self.canvas.set_ui_font_size(self.ui_font_size)
        self.apply_button_density()
        self.updateGeometry()
        self.update()
        self.save_settings()

    def change_font_size(self, delta):
        self.apply_ui_font_size(self.ui_font_size + delta)
        self.status.showMessage(f"Font size set to {self.ui_font_size} pt.", 3000)

    def reset_font_size(self):
        default_size = QFont().pointSize()
        if default_size <= 0:
            default_size = 9
        self.apply_ui_font_size(default_size)
        self.status.showMessage(f"Font size reset to {self.ui_font_size} pt.", 3000)

    def update_autosave_controls_visibility(self):
        timed_mode = (self.autosave_mode.currentData() or "image_change") == "timed"
        self.autosave_minutes.setVisible(timed_mode)

    def apply_button_density(self):
        button_height = max(28, self.ui_font_size * 3)
        primary_height = max(34, self.ui_font_size * 3 + 2)
        for button in getattr(self, "action_buttons", {}).values():
            button.setMinimumHeight(primary_height if button.objectName() == "primaryButton" else button_height)

    def on_auto_next_mode_changed(self, _index):
        self.auto_advance_mode = self.auto_next_mode.currentData() or "none"
        mode_text = {
            "none": "Next image auto-annotate is disabled.",
            "full": "Next image will auto-suggest on the whole image.",
            "roi": "Next image will auto-suggest inside the previous ROI.",
        }.get(self.auto_advance_mode, "Next image behavior updated.")
        self.status.showMessage(mode_text, 3000)

    def show_help_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Help")
        dialog.resize(640, 560)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "\n".join(
                [
                    "Workspace:",
                    "- Open a folder with images and YOLO txt labels.",
                    "- Classes are loaded from classes.txt or obj.names.",
                    "- Sort images by name, labeled state, or reviewed state from the left panel.",
                    "- Project/Review tabs: Ctrl+1 (Project), Ctrl+2 (Review + Auto).",
                    "",
                    "Annotation:",
                    "- Draw with left drag.",
                    "- Hold B while clicking inside an existing box to force drawing a new box there.",
                    "- Move selected boxes by drag or Ctrl+arrow keys.",
                    "- Ctrl+drag a selected box to move all selected boxes together.",
                    "- Ctrl+Click boxes to multi-select them.",
                    "- Ctrl+A selects all labels on the current image.",
                    "- Ctrl+C copies selected labels, Ctrl+X cuts, Ctrl+V pastes.",
                    "- Ctrl+Shift+N copies current labels to next image as suggestions (if next is unlabeled).",
                    "- Hold Space and drag to pan.",
                    "- Z undo, Y redo, C clears labels (ROI-only if ROI active).",
                    "",
                    "ROI:",
                    "- Enable ROI select mode, then drag a free lasso region.",
                    "- Finishing an ROI selects existing labels inside that region.",
                    "- ROI applies to Accept/Accept All (accepts suggestions inside ROI).",
                    "- X rejects suggestions (ROI-only if ROI active; otherwise all).",
                    "",
                    "Auto Annotate:",
                    "- Auto Suggest creates suggestions first; drag a suggestion to accept it.",
                    "- Select classes under Suggest Classes to limit which classes are suggested.",
                    "- Model mapped X/Y means X model classes matched your project out of Y total.",
                    "- Unmapped model classes are skipped to avoid writing wrong class IDs.",
                    "- Auto-import model classes adds missing selected model classes before suggesting (toggle).",
                    "",
                    "Review:",
                    "- Enter accepts selected suggestion.",
                    "- Ctrl+Enter accepts all suggestions (ROI-only if ROI active).",
                    "- Delete rejects a selected suggestion during review.",
                    "- R toggles reviewed.",
                    "- W jumps to the next unlabeled image.",
                    "- A/D or Left/Right moves prev/next image.",
                    "",
                    "Settings:",
                    "- Use File/Tools -> Configure Shortcuts to change shortcuts.",
                    "- Use Tools -> Increase/Decrease Font Size to adjust text size.",
                    "- Config import/export stores UI settings, shortcuts, and font size.",
                ]
            )
        )
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About",
            "YOLO Advanced Annotator\n\nCreated by Arnob\nEnhanced with review tools, ROI auto-labeling, and persistent desktop workflow features.",
        )

    def current_image_name(self):
        if 0 <= self.current_image_index < len(self.image_files):
            return self.image_files[self.current_image_index]
        return ""

    def current_image_path(self):
        name = self.current_image_name()
        if self.folder and name:
            return os.path.join(self.folder, name)
        return ""

    def sync_images_with_disk(self, reload_current=True, notify=False, force=False):
        if not self.folder or not os.path.isdir(self.folder):
            return False
        now = time.monotonic()
        if not force and now - self.last_disk_sync_time < 1.0:
            return False
        self.last_disk_sync_time = now
        current_name = self.current_image_name()
        scanned = [entry.name for entry in os.scandir(self.folder) if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS)]
        if set(scanned) == set(self.image_files) and len(scanned) == len(self.image_files):
            return False

        previous_index = self.current_image_index
        self.image_files = scanned
        self.refresh_dataset_cache()
        self.apply_sort_mode()
        self.refresh_file_list(preserve_row=False)

        removed_current = current_name and current_name not in self.image_files
        if not self.image_files:
            self.current_image_index = -1
            self.canvas.clear_loaded_image()
            self.image_info.setText("No images in folder")
            self.refresh_summary_labels()
            if notify:
                self.status.showMessage("All images are missing from the folder.", 5000)
            return True

        if current_name in self.image_files:
            new_index = self.image_files.index(current_name)
        else:
            new_index = min(max(previous_index, 0), len(self.image_files) - 1)

        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(new_index)
        self.file_list.blockSignals(False)
        self.current_image_index = new_index
        if removed_current:
            self.skip_save_on_next_load = True
            self.canvas.clear_loaded_image()
        if reload_current:
            self.load_current_image(new_index)
        if removed_current and notify:
            self.status.showMessage(f"Image removed from folder: {current_name}", 5000)
        elif notify:
            self.status.showMessage("Folder contents changed. Image list was refreshed.", 4000)
        return True

    def label_path_for_image(self, image_name):
        return os.path.splitext(os.path.join(self.folder, image_name))[0] + ".txt"

    def reviewed_path_for_image(self, image_name):
        return os.path.join(self.folder, image_name + REVIEW_SUFFIX)

    def has_saved_labels(self, image_name):
        if image_name not in self.label_cache:
            self.label_cache[image_name] = self.compute_label_state(image_name)
        return self.label_cache[image_name]

    def is_reviewed_image(self, image_name):
        if image_name not in self.reviewed_cache:
            self.reviewed_cache[image_name] = self.compute_reviewed_state(image_name)
        return self.reviewed_cache[image_name]

    def update_counts_for_label_state(self, image_name, new_state):
        old_state = self.label_cache.get(image_name)
        if old_state is None:
            old_state = self.compute_label_state(image_name)
        if bool(old_state) != bool(new_state):
            self.labeled_count += 1 if new_state else -1
        self.label_cache[image_name] = bool(new_state)

    def update_counts_for_reviewed_state(self, image_name, new_state):
        old_state = self.reviewed_cache.get(image_name)
        if old_state is None:
            old_state = self.compute_reviewed_state(image_name)
        if bool(old_state) != bool(new_state):
            self.reviewed_count += 1 if new_state else -1
        self.reviewed_cache[image_name] = bool(new_state)

    def refresh_summary_labels(self):
        self.dataset_info.setText(
            f"Dataset: {len(self.image_files)} images | labeled {self.labeled_count} | reviewed {self.reviewed_count}"
        )

    def update_file_list_item(self, image_name):
        try:
            row = self.image_files.index(image_name)
        except ValueError:
            return
        item = self.file_list.item(row)
        if item is not None:
            item.setText(self.decorate_image_name(image_name))

    def decorate_image_name(self, image_name):
        flags = []
        if self.has_saved_labels(image_name):
            flags.append("L")
        if self.is_reviewed_image(image_name):
            flags.append("R")
        prefix = f"[{' '.join(flags)}] " if flags else ""
        return prefix + image_name

    def refresh_file_list(self, preserve_row=True):
        selected_row = self.file_list.currentRow() if preserve_row else -1
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for image_name in self.image_files:
            self.file_list.addItem(self.decorate_image_name(image_name))
        if self.image_files:
            if selected_row < 0:
                selected_row = min(max(self.current_image_index, 0), len(self.image_files) - 1)
            self.file_list.setCurrentRow(min(selected_row, len(self.image_files) - 1))
        self.file_list.blockSignals(False)
        self.refresh_summary_labels()

    def label_counts_for_images(self, image_names):
        counts = {}
        total_labels = 0
        for image_name in image_names:
            summary = self.get_label_summary(image_name)
            for annotation in summary["annotations"]:
                counts[annotation.class_id] = counts.get(annotation.class_id, 0) + 1
                total_labels += 1
        return counts, total_labels

    def invalidate_label_summary(self, image_name):
        if image_name:
            self.label_summary_cache.pop(image_name, None)

    def get_label_summary(self, image_name):
        label_path = self.label_path_for_image(image_name)
        exists = os.path.exists(label_path)
        mtime = os.path.getmtime(label_path) if exists else None
        cached = self.label_summary_cache.get(image_name)
        if cached and cached["mtime"] == mtime:
            return cached
        annotations = self.read_annotations_for_image(image_name) if exists else []
        counts = {}
        for annotation in annotations:
            counts[annotation.class_id] = counts.get(annotation.class_id, 0) + 1
        summary = {
            "mtime": mtime,
            "annotations": annotations,
            "counts": counts,
            "total": len(annotations),
        }
        self.label_summary_cache[image_name] = summary
        return summary

    def get_image_features(self, image_path):
        try:
            mtime = os.path.getmtime(image_path)
        except OSError:
            return None
        cached = self.image_feature_cache.get(image_path)
        if cached and cached["mtime"] == mtime:
            return cached
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        try:
            orb = cv2.ORB_create(nfeatures=2000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
        except cv2.error:
            return None
        features = {
            "mtime": mtime,
            "gray": gray,
            "keypoints": keypoints or [],
            "descriptors": descriptors,
        }
        self.image_feature_cache[image_path] = features
        if len(self.image_feature_cache) > MAX_FEATURE_CACHE:
            oldest_key = next(iter(self.image_feature_cache))
            self.image_feature_cache.pop(oldest_key, None)
        return features

    def get_image_size(self, image_name):
        image_path = os.path.join(self.folder, image_name)
        try:
            mtime = os.path.getmtime(image_path)
        except OSError:
            return None
        cached = self.image_size_cache.get(image_path)
        if cached and cached["mtime"] == mtime:
            return cached["size"]
        image = cv2.imread(image_path)
        if image is None:
            return None
        size = image.shape[:2]
        self.image_size_cache[image_path] = {"mtime": mtime, "size": size}
        return size

    def format_label_counts(self, counts):
        if not counts:
            return "none"
        parts = []
        for class_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            parts.append(f"{self.canvas.class_name(class_id)}={count}")
        return ", ".join(parts)

    def read_class_names_from_file(self, path):
        if not path or not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def merge_preserving_existing_ids(self, base_classes, extra_classes):
        merged = list(base_classes)
        existing = {self.normalize_class_name(name) for name in merged}
        for class_name in extra_classes:
            normalized = self.normalize_class_name(class_name)
            if normalized not in existing:
                merged.append(class_name)
                existing.add(normalized)
        return merged

    def load_classes(self):
        self.class_file_path = ""
        backup_path = os.path.join(self.folder, "classes.project_backup.txt")
        backup_classes = self.read_class_names_from_file(backup_path)
        disk_classes = []
        for name in CLASS_FILE_NAMES:
            candidate = os.path.join(self.folder, name)
            if os.path.exists(candidate):
                self.class_file_path = candidate
                disk_classes = self.read_class_names_from_file(candidate)
                break
        if disk_classes:
            classes = self.merge_preserving_existing_ids(disk_classes, backup_classes)
        elif backup_classes:
            classes = list(backup_classes)
        else:
            classes = list(disk_classes)
        if not classes:
            classes = ["object"]
        self.project_master_classes = list(classes)
        self.class_file_path = os.path.join(self.folder, "classes.txt")
        self.save_classes(self.project_master_classes)
        self.populate_classes(list(self.project_master_classes))
        self.backup_project_classes()

    def save_classes(self, classes=None):
        if not self.folder:
            return
        classes = classes if classes is not None else (self.project_master_classes or self.canvas.classes)
        classes_path = os.path.join(self.folder, "classes.txt")
        atomic_write_lines(classes_path, [class_name + "\n" for class_name in classes])
        obj_names_path = os.path.join(self.folder, "obj.names")
        if self.class_file_path and self.class_file_path.endswith("obj.names") or os.path.exists(obj_names_path):
            atomic_write_lines(obj_names_path, [class_name + "\n" for class_name in classes])
        self.class_file_path = classes_path

    def backup_project_classes(self, force=False):
        classes = self.project_master_classes or self.canvas.classes
        if not self.folder or not classes:
            return
        backup_path = os.path.join(self.folder, "classes.project_backup.txt")
        if os.path.exists(backup_path) and not force:
            return
        atomic_write_lines(backup_path, [class_name + "\n" for class_name in classes])

    def populate_classes(self, classes):
        self.class_list.blockSignals(True)
        self.class_picker.blockSignals(True)
        self.class_list.clear()
        self.class_picker.clear()
        self.canvas.set_classes(classes)
        for index, class_name in enumerate(classes):
            text = f"{class_name} ({index})"
            self.class_picker.addItem(text)
        items = list(enumerate(classes))
        if (self.class_sort_mode.currentData() or "id") == "az":
            items = sorted(items, key=lambda item: item[1].lower())
        self.class_list_row_by_id = {}
        for row_index, (class_id, class_name) in enumerate(items):
            text = f"{class_name} ({class_id})"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, class_id)
            self.class_list.addItem(item)
            self.class_list_row_by_id[class_id] = row_index
        self.class_list.blockSignals(False)
        self.class_picker.blockSignals(False)
        self.refresh_suggest_class_filter()
        self.select_class_by_index(0)
        if self.detector:
            self.build_model_class_mapping()

    def refresh_suggest_class_filter(self):
        selected_names = {item.text() for item in self.suggest_class_filter.selectedItems()}
        if self.saved_suggest_class_names:
            selected_names = set(self.saved_suggest_class_names)
        self.suggest_class_filter.blockSignals(True)
        self.suggest_class_filter.clear()
        if self.detector and self.model_class_names:
            class_source = list(self.model_class_names)
        else:
            class_source = list(self.canvas.classes)
        for class_name in class_source:
            item = QListWidgetItem(class_name)
            self.suggest_class_filter.addItem(item)
            if class_name in selected_names:
                item.setSelected(True)
        self.suggest_class_filter.blockSignals(False)
        self.saved_suggest_class_names = []

    def selected_suggest_class_ids(self):
        selected_items = self.suggest_class_filter.selectedItems()
        if not selected_items:
            return None
        selected_names = {item.text() for item in selected_items}
        if self.detector and self.model_class_names:
            allowed = set()
            for model_index, class_name in enumerate(self.model_class_names):
                if class_name in selected_names:
                    mapped = self.model_class_map.get(model_index)
                    if mapped is not None:
                        allowed.add(mapped)
            return allowed if allowed else None
        return {index for index, class_name in enumerate(self.canvas.classes) if class_name in selected_names}

    def normalized_roi_from_canvas(self):
        if not self.canvas.has_roi() or not self.canvas.has_image():
            return []
        h, w = self.canvas.image.shape[:2]
        return [(x / max(w, 1), y / max(h, 1)) for x, y in self.canvas.roi_points]

    def apply_persisted_roi_to_canvas(self):
        if not self.persist_roi_across_images or not self.normalized_roi_points or not self.canvas.has_image():
            return
        h, w = self.canvas.image.shape[:2]
        self.canvas.roi_points = [(nx * w, ny * h) for nx, ny in self.normalized_roi_points]
        self.canvas.roi_preview_point = None
        self.update_roi_status(self.canvas.roi_bounds_text())
        self.canvas.update()

    def set_persist_roi(self, checked):
        self.persist_roi_across_images = checked
        if checked:
            self.normalized_roi_points = self.normalized_roi_from_canvas()
        else:
            self.normalized_roi_points = []

    def set_keyboard_review_mode(self, checked):
        self.keyboard_review_mode = checked
        self.status.showMessage(
            "Keyboard review mode enabled: use A/D, U, X, Enter, Ctrl+Enter, [ and ]"
            if checked
            else "Keyboard review mode disabled",
            3000,
        )

    def annotations_from_recent_history(self):
        if not self.previous_annotations_cache:
            return []
        return [Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score) for item in self.previous_annotations_cache]

    def load_current_image(self, row):
        if row < 0 or row >= len(self.image_files):
            return
        previous_index = self.current_image_index
        if self.persist_roi_across_images:
            self.normalized_roi_points = self.normalized_roi_from_canvas()
        if self.skip_save_on_next_load:
            self.skip_save_on_next_load = False
        elif self.autosave_on_image_change() and self.save_current_labels() is False:
            return
        if self.canvas.annotations:
            self.previous_annotations_cache = [
                Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score) for item in self.canvas.annotations
            ]
        self.current_image_index = row
        self.replace_pending = False
        path = self.current_image_path()
        if not path or not os.path.exists(path):
            self.sync_images_with_disk(reload_current=True, notify=True, force=True)
            return
        self.loading_image = True
        try:
            self.canvas.load_image(path)
        except ValueError as exc:
            self.current_image_index = previous_index
            self.file_list.blockSignals(True)
            if 0 <= previous_index < self.file_list.count():
                self.file_list.setCurrentRow(previous_index)
            self.file_list.blockSignals(False)
            self.loading_image = False
            QMessageBox.warning(self, "Image load failed", str(exc))
            return
        self.loading_image = False
        if self.persist_roi_across_images:
            self.apply_persisted_roi_to_canvas()
        else:
            self.canvas.clear_roi()
        self.image_info.setText(f"{row + 1}/{len(self.image_files)}  {self.image_files[row]}")
        self.reviewed_check.blockSignals(True)
        self.reviewed_check.setChecked(self.is_reviewed_image(self.image_files[row]))
        self.reviewed_check.blockSignals(False)
        self.refresh_annotation_list()
        self.set_current_labels_clean(self.image_files[row])
        if self.pending_next_suggestion_image and self.pending_next_suggestion_image == self.current_image_name():
            if self.canvas.annotations:
                self.status.showMessage("Next-image suggestions skipped because labels already exist.", 4000)
            elif self.pending_next_suggestions:
                h, w = self.canvas.image.shape[:2]
                suggestions = []
                for item in self.pending_next_suggestions:
                    clone = Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score)
                    clone.clamp(w, h)
                    suggestions.append(clone)
                self.canvas.set_suggestions(suggestions)
                self.status.showMessage(f"Copied {len(suggestions)} labels as suggestions for this image.", 4000)
            self.pending_next_suggestions = None
            self.pending_next_suggestion_image = ""
            self.pending_next_suggestion_count = 0
        elif self.auto_next_suggest_check.isChecked() and self.previous_annotations_cache:
            h, w = self.canvas.image.shape[:2]
            detections = []
            for item in self.previous_annotations_cache:
                clone = Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score)
                clone.clamp(w, h)
                detections.append((clone.class_id, clone.x1, clone.y1, clone.x2, clone.y2, clone.score))
            existing_suggestions = list(self.canvas.suggestions)
            base_annotations = list(self.canvas.annotations) + existing_suggestions
            suggestions = self.filter_new_detections(detections, base_annotations, None, None)
            if suggestions:
                self.canvas.set_suggestions(existing_suggestions + suggestions)
                self.status.showMessage(f"Suggested {len(suggestions)} labels from previous image.", 3000)
        if self.auto_advance_mode == "full":
            self.canvas.clear_roi()
        self.maybe_auto_annotate_loaded_image()
        self.save_settings()

    def save_current_labels(self):
        if not self.canvas.has_image():
            return True
        if not self.labels_dirty:
            self.autosave_timer.stop()
            self.update_autosave_status()
            return True
        image_path = self.current_image_path()
        if not image_path or not os.path.exists(image_path):
            self.sync_images_with_disk(reload_current=True, notify=True, force=True)
            QMessageBox.warning(self, "Image missing", "The current image was removed from disk. Labels were not saved.")
            return False
        self.autosave_timer.stop()
        try:
            self.canvas.save_labels()
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save labels: {exc}")
            return False
        image_name = self.current_image_name()
        if image_name:
            self.update_counts_for_label_state(image_name, len(self.canvas.annotations) > 0)
            self.invalidate_label_summary(image_name)
            self.update_file_list_item(image_name)
            self.set_current_labels_clean(image_name, update_status=False)
        self.refresh_summary_labels()
        self.mark_autosave_saved("manual")
        self.status.showMessage(f"Saved {os.path.basename(self.canvas.label_path())}", 3000)
        return True

    def schedule_autosave(self):
        if not self.labels_dirty:
            self.autosave_timer.stop()
            self.update_autosave_status()
            return
        if self.autosave.isChecked() and self.canvas.has_image() and (self.autosave_mode.currentData() or "image_change") == "timed":
            self.autosave_timer.start(int(max(self.autosave_minutes.value(), 0.1) * 60 * 1000))
            self.update_autosave_status(True)

    def autosave_current_labels(self):
        if not self.autosave.isChecked() or not self.canvas.has_image():
            return True
        if not self.labels_dirty:
            self.autosave_timer.stop()
            self.update_autosave_status()
            return True
        image_path = self.current_image_path()
        if not image_path or not os.path.exists(image_path):
            self.sync_images_with_disk(reload_current=True, notify=True, force=True)
            self.status.showMessage("Autosave skipped because the current image is missing from disk.", 5000)
            return False
        self.autosave_timer.stop()
        try:
            self.canvas.save_labels()
        except Exception as exc:
            self.status.showMessage(f"Autosave failed: {exc}", 5000)
            return False
        image_name = self.current_image_name()
        if image_name:
            self.update_counts_for_label_state(image_name, len(self.canvas.annotations) > 0)
            self.invalidate_label_summary(image_name)
            self.update_file_list_item(image_name)
            self.set_current_labels_clean(image_name, update_status=False)
        self.refresh_summary_labels()
        self.mark_autosave_saved("autosave")
        return True

    def refresh_annotation_list(self):
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for index, annotation in enumerate(self.canvas.annotations):
            box = annotation.normalized()
            class_name = self.canvas.class_name(annotation.class_id)
            text = f"{index + 1}. {class_name}  ({int(box.x1)}, {int(box.y1)}) - ({int(box.x2)}, {int(box.y2)})"
            item = QListWidgetItem(text)
            accent = self.canvas.annotation_color(annotation.class_id)
            item.setForeground(QColor(25, 25, 25))
            item.setBackground(QColor(accent.red(), accent.green(), accent.blue(), 45))
            self.annotation_list.addItem(item)
        self.annotation_list.clearSelection()
        selected = set(self.canvas.selected_indices)
        if not selected and 0 <= self.canvas.selected_index < self.annotation_list.count():
            selected = {self.canvas.selected_index}
        for row in sorted(selected):
            item = self.annotation_list.item(row)
            if item is not None:
                item.setSelected(True)
        if 0 <= self.canvas.selected_index < self.annotation_list.count():
            self.annotation_list.setCurrentRow(self.canvas.selected_index)
        self.annotation_list.blockSignals(False)
        self.schedule_autosave()
        self.refresh_suggestion_info()

    def refresh_suggestion_info(self):
        count = len(self.canvas.suggestions)
        info_text = f"Suggestions: {count}"
        self.suggestion_info.setText(info_text)
        if self.suggestion_list.isHidden():
            return
        self.suggestion_list.blockSignals(True)
        self.suggestion_list.clear()
        for index, suggestion in enumerate(self.canvas.suggestions):
            class_name = self.canvas.class_name(suggestion.class_id)
            score_text = f"{suggestion.score:.2f}" if suggestion.score >= 0 else "--"
            box = suggestion.normalized()
            item = QListWidgetItem(
                f"{index + 1}. {class_name} | conf {score_text} | ({int(box.x1)}, {int(box.y1)}) - ({int(box.x2)}, {int(box.y2)})"
            )
            item.setForeground(QColor(0, 180, 220))
            self.suggestion_list.addItem(item)
        if 0 <= self.canvas.selected_suggestion_index < self.suggestion_list.count():
            self.suggestion_list.setCurrentRow(self.canvas.selected_suggestion_index)
        self.suggestion_list.blockSignals(False)

    def update_roi_status(self, text):
        self.roi_info.setText(text or "ROI: full image")
        self.roi_mode_btn.blockSignals(True)
        self.roi_mode_btn.setChecked(self.canvas.roi_mode)
        self.roi_mode_btn.blockSignals(False)

    def on_suggestion_clicked(self, row):
        if row >= 0:
            self.canvas.selected_suggestion_index = row
            self.canvas.set_selected_index(-1)
            self.canvas.suggestions_changed.emit()
            self.canvas.update()

    def sync_annotation_selection(self, index):
        if not self.syncing_annotation_list:
            self.syncing_annotation_list = True
            self.annotation_list.blockSignals(True)
            self.annotation_list.clearSelection()
            selected = set(self.canvas.selected_indices)
            if not selected and index >= 0:
                selected = {index}
            for row in sorted(selected):
                item = self.annotation_list.item(row)
                if item is not None:
                    item.setSelected(True)
            if index >= 0:
                self.annotation_list.setCurrentRow(index)
            self.annotation_list.blockSignals(False)
            self.syncing_annotation_list = False
        if index >= 0:
            self.canvas.selected_suggestion_index = -1
            self.refresh_suggestion_info()
        annotation = self.canvas.selected_annotation()
        if annotation:
            self.class_picker.blockSignals(True)
            self.class_picker.setCurrentIndex(annotation.class_id)
            self.class_picker.blockSignals(False)
            self.class_list.blockSignals(True)
            self.class_list.setCurrentRow(self.class_list_row_by_id.get(annotation.class_id, annotation.class_id))
            self.class_list.blockSignals(False)

    def on_annotation_selection_changed(self):
        if self.syncing_annotation_list:
            return
        selected_rows = {self.annotation_list.row(item) for item in self.annotation_list.selectedItems()}
        self.syncing_annotation_list = True
        if selected_rows:
            self.canvas.set_selected_indices(selected_rows)
        else:
            self.canvas.set_selected_index(-1)
        self.syncing_annotation_list = False

    def on_class_selected(self, row):
        if row >= 0:
            item = self.class_list.item(row)
            class_id = item.data(Qt.UserRole) if item is not None else row
            if class_id is None:
                class_id = row
            class_id = int(class_id)
            self.class_picker.setCurrentIndex(class_id)
            self.canvas.current_class_id = class_id
            if self.canvas.selected_indices:
                valid = [index for index in self.canvas.selected_indices if 0 <= index < len(self.canvas.annotations)]
                for index in valid:
                    self.canvas.annotations[index].class_id = class_id
                self.canvas.annotations_changed.emit()
                self.canvas.update()

    def on_class_picker_changed(self, index):
        if index >= 0:
            target_row = self.class_list_row_by_id.get(index, index)
            self.class_list.setCurrentRow(target_row)
            self.canvas.current_class_id = index
            if self.canvas.selected_indices:
                valid = [row for row in self.canvas.selected_indices if 0 <= row < len(self.canvas.annotations)]
                for row in valid:
                    self.canvas.annotations[row].class_id = index
                self.canvas.annotations_changed.emit()
                self.canvas.update()

    def on_class_sort_changed(self, _index):
        self.populate_classes(list(self.canvas.classes))

    def make_unique_class_name(self, base_name, classes, exclude_index=None):
        used = {name.strip().lower() for index, name in enumerate(classes) if index != exclude_index}
        candidate = base_name.strip()
        if candidate.lower() not in used:
            return candidate
        counter = 2
        while True:
            candidate = f"{base_name}_{counter}"
            if candidate.lower() not in used:
                return candidate
            counter += 1

    def find_class_id_by_name(self, name, classes, exclude_index=None):
        target = name.strip().lower()
        for index, existing in enumerate(classes):
            if index == exclude_index:
                continue
            if existing.strip().lower() == target:
                return index
        return None

    def remove_class_by_id(self, class_id, remap_to=None):
        return self.remove_class_by_id_global(class_id, remap_to, apply_dataset=True)

    def remove_class_by_id_global(self, class_id, remap_to=None, apply_dataset=False):
        classes = list(self.project_master_classes or self.canvas.classes)
        if class_id < 0 or class_id >= len(classes):
            return None
        removed_name = classes.pop(class_id)
        self.project_master_classes = list(classes)
        self.populate_classes(classes)
        self.save_classes(self.project_master_classes)
        self.backup_project_classes(force=True)

        if apply_dataset and self.folder:
            self.remap_labels_in_dataset(class_id, remap_to)

        self.select_class_by_index(min(class_id, len(classes) - 1) if classes else 0)
        self.canvas.set_selected_index(-1)
        self.canvas.annotations_changed.emit()
        self.canvas.update()
        return removed_name

    def remap_labels_in_dataset(self, removed_id, remap_to=None):
        return self.remap_labels_in_dataset_internal(removed_id, remap_to, dry_run=False)

    def remap_labels_in_dataset_internal(self, removed_id, remap_to=None, dry_run=False):
        summary = {
            "files_scanned": 0,
            "files_updated": 0,
            "labels_removed": 0,
            "labels_remapped": 0,
            "labels_shifted": 0,
            "errors": [],
        }
        if not self.folder:
            return summary
        class_limit = len(self.canvas.classes)
        progress = None
        if not dry_run:
            progress = QProgressDialog("Updating labels...", "", 0, max(len(self.image_files), 1), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButtonText("Cancel")
            progress.setMinimumDuration(250)
            progress.show()
            self.setEnabled(False)
        try:
            for index, image_name in enumerate(self.image_files, start=1):
                if progress:
                    progress.setValue(index - 1)
                    progress.setLabelText(f"Updating {image_name} ({index}/{len(self.image_files)})")
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        summary["errors"].append("Operation canceled by user.")
                        break
                label_path = self.label_path_for_image(image_name)
                if not os.path.exists(label_path):
                    continue
                summary["files_scanned"] += 1
                updated = False
                new_lines = []
                with open(label_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        if class_id == removed_id:
                            updated = True
                            if remap_to is None:
                                summary["labels_removed"] += 1
                                continue
                            target_id = remap_to - 1 if remap_to > removed_id else remap_to
                            class_id = max(0, min(target_id, class_limit - 1))
                            summary["labels_remapped"] += 1
                        elif class_id > removed_id:
                            class_id -= 1
                            updated = True
                            summary["labels_shifted"] += 1
                        new_lines.append(f"{class_id} {' '.join(parts[1:])}\n")
                if updated:
                    summary["files_updated"] += 1
                    if not dry_run:
                        try:
                            if new_lines:
                                atomic_write_lines(label_path, new_lines)
                            else:
                                os.remove(label_path)
                        except OSError as exc:
                            summary["errors"].append(f"{os.path.basename(label_path)}: {exc}")
                            continue
                        self.invalidate_label_summary(image_name)
                        self.update_counts_for_label_state(image_name, len(new_lines) > 0)
                        self.update_file_list_item(image_name)
        finally:
            if progress:
                progress.setValue(max(len(self.image_files), 1))
                progress.close()
                self.setEnabled(True)

        if not dry_run and self.canvas.has_image():
            updated_annotations = []
            for annotation in self.canvas.annotations:
                if annotation.class_id == removed_id:
                    if remap_to is None:
                        continue
                    annotation.class_id = remap_to - 1 if remap_to > removed_id else remap_to
                elif annotation.class_id > removed_id:
                    annotation.class_id -= 1
                updated_annotations.append(annotation)
            self.canvas.annotations = updated_annotations
            self.canvas.annotations_changed.emit()
            self.canvas.update()
            self.refresh_summary_labels()
        return summary

    def confirm_dataset_class_change(self, action_label, removed_id, remap_to=None, removed_name="", target_name=""):
        summary = self.remap_labels_in_dataset_internal(removed_id, remap_to, dry_run=True)
        extra = ""
        if target_name:
            extra = f" into '{target_name}'"
        message = (
            f"{action_label} '{removed_name}'{extra} across the dataset?\n\n"
            f"Files scanned: {summary['files_scanned']}\n"
            f"Files updated: {summary['files_updated']}\n"
            f"Labels removed: {summary['labels_removed']}\n"
            f"Labels remapped: {summary['labels_remapped']}\n"
            f"Labels shifted: {summary['labels_shifted']}\n\n"
            "This will update all label files."
        )
        answer = QMessageBox.question(self, action_label, message)
        if answer != QMessageBox.Yes:
            return False
        return self.backup_labels_before_change()

    def backup_labels_before_change(self):
        if not self.folder:
            return True
        backup_root = os.path.join(self.folder, "_label_backups")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(backup_root, f"backup_{timestamp}")
        try:
            os.makedirs(backup_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Backup failed", f"Could not create backup folder: {exc}")
            return False
        copied = 0
        errors = []
        for image_name in self.image_files:
            label_path = self.label_path_for_image(image_name)
            if not os.path.exists(label_path):
                continue
            target_path = os.path.join(backup_dir, os.path.basename(label_path))
            try:
                with open(label_path, "rb") as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                copied += 1
            except OSError as exc:
                errors.append(f"{os.path.basename(label_path)}: {exc}")
        if errors:
            QMessageBox.warning(
                self,
                "Backup warning",
                "Some labels could not be backed up:\n" + "\n".join(errors[:8]),
            )
        self.status.showMessage(f"Backed up {copied} label files to {backup_dir}", 5000)
        return True

    def add_class(self):
        name, ok = QInputDialog.getText(self, "Add Class", "Class name")
        if not ok or not name.strip():
            return
        classes = list(self.project_master_classes or self.canvas.classes) + [name.strip()]
        self.project_master_classes = list(classes)
        self.populate_classes(classes)
        self.save_classes(self.project_master_classes)
        self.backup_project_classes(force=True)
        self.select_class_by_index(len(classes) - 1)

    def rename_class(self):
        item = self.class_list.currentItem()
        if item is None:
            return
        class_id = item.data(Qt.UserRole)
        if class_id is None:
            class_id = self.class_list.currentRow()
        class_id = int(class_id)
        if class_id < 0 or class_id >= len(self.canvas.classes):
            return
        current_name = self.canvas.classes[class_id]
        name, ok = QInputDialog.getText(self, "Rename Class", "Class name", text=current_name)
        if not ok or not name.strip():
            return
        new_name = name.strip()
        classes = list(self.project_master_classes or self.canvas.classes)

        if new_name.lower().startswith("dup_"):
            target_name = new_name[4:].strip()
            target_id = self.find_class_id_by_name(target_name, classes, exclude_index=class_id)
            if target_id is not None and target_id != class_id:
                display_target = classes[target_id]
                if self.confirm_dataset_class_change("Merge Classes", class_id, target_id, current_name, display_target):
                    removed_name = self.remove_class_by_id_global(class_id, remap_to=target_id, apply_dataset=True)
                    if removed_name:
                        self.status.showMessage(f"Merged {removed_name} into {display_target}", 4000)
                    return

        duplicate_id = self.find_class_id_by_name(new_name, classes, exclude_index=class_id)
        if duplicate_id is not None:
            if current_name.strip().lower().startswith("dup_"):
                target_name = classes[duplicate_id]
                if self.confirm_dataset_class_change("Merge Classes", class_id, duplicate_id, current_name, target_name):
                    removed_name = self.remove_class_by_id_global(class_id, remap_to=duplicate_id, apply_dataset=True)
                    if removed_name:
                        self.status.showMessage(f"Merged {removed_name} into {target_name}", 4000)
                    return
            answer = QMessageBox.question(
                self,
                "Duplicate Class",
                f"A class named '{new_name}' already exists.\nRename this class to 'dup_{new_name}' instead?",
            )
            if answer != QMessageBox.Yes:
                return
            new_name = self.make_unique_class_name(f"dup_{new_name}", classes, exclude_index=class_id)

        classes[class_id] = new_name
        self.project_master_classes = list(classes)
        self.populate_classes(classes)
        self.save_classes(self.project_master_classes)
        self.backup_project_classes(force=True)
        self.select_class_by_index(class_id)
        self.refresh_annotation_list()

    def delete_class(self):
        if len(self.canvas.classes) <= 1:
            QMessageBox.information(self, "Class required", "At least one class must remain.")
            return
        item = self.class_list.currentItem()
        if item is None:
            return
        class_id = item.data(Qt.UserRole)
        if class_id is None:
            class_id = self.class_list.currentRow()
        class_id = int(class_id)
        if class_id < 0 or class_id >= len(self.canvas.classes):
            return
        classes = list(self.project_master_classes or self.canvas.classes)
        current_name = classes[class_id]
        if not current_name.lower().startswith("del_"):
            renamed = self.make_unique_class_name(f"del_{current_name}", classes, exclude_index=class_id)
            classes[class_id] = renamed
            self.project_master_classes = list(classes)
            self.populate_classes(classes)
            self.save_classes(self.project_master_classes)
            self.backup_project_classes(force=True)
            self.select_class_by_index(class_id)
            self.canvas.update()
            self.status.showMessage(
                f"Class renamed to {renamed}. Press Delete again to permanently remove it.", 4000
            )
            return

        if self.confirm_dataset_class_change("Delete Class", class_id, None, current_name):
            removed_name = self.remove_class_by_id_global(class_id, apply_dataset=True)
            if removed_name:
                self.status.showMessage(f"Deleted class {removed_name}", 3000)

    def toggle_reviewed(self, checked):
        image_name = self.current_image_name()
        if not image_name:
            return
        marker_path = self.reviewed_path_for_image(image_name)
        try:
            if checked:
                atomic_write_lines(marker_path, ["reviewed\n"])
            elif os.path.exists(marker_path):
                os.remove(marker_path)
        except OSError as exc:
            QMessageBox.warning(self, "Reviewed state failed", f"Could not update reviewed marker: {exc}")
            self.reviewed_check.blockSignals(True)
            self.reviewed_check.setChecked(not checked)
            self.reviewed_check.blockSignals(False)
            return
        self.update_counts_for_reviewed_state(image_name, checked)
        self.update_file_list_item(image_name)
        self.refresh_summary_labels()

    def load_model_file(self):
        if YOLO is None:
            QMessageBox.warning(
                self,
                "Missing dependency",
                "Semi-annotation needs the ultralytics package. Install it with: pip install ultralytics",
            )
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO .pt model", self.folder or os.getcwd(), "PyTorch Model (*.pt)")
        if not path:
            return
        try:
            self.detector = YOLO(path)
            self.model_path = path
            self.build_model_class_mapping()
            self.save_settings()
        except Exception as exc:
            QMessageBox.warning(self, "Model load failed", str(exc))

    def unload_model(self):
        self.detector = None
        self.model_path = ""
        self.model_class_names = []
        self.model_class_map = {}
        self.model_info.setText("Model: none")
        self.refresh_suggest_class_filter()
        self.refresh_summary_labels()
        self.save_settings()
        self.status.showMessage("Model unloaded.", 3000)

    def build_model_class_mapping(self):
        if not self.detector:
            return
        names = getattr(self.detector.model, "names", None)
        if isinstance(names, dict) and names:
            model_classes = [str(names[idx]) for idx in sorted(names)]
        elif isinstance(names, list) and names:
            model_classes = [str(name) for name in names]
        else:
            self.model_class_names = []
            self.model_class_map = {}
            self.model_info.setText(f"Model: {os.path.basename(self.model_path)}")
            self.status.showMessage(f"Loaded model {os.path.basename(self.model_path)}", 4000)
            return
        if not (self.project_master_classes or self.canvas.classes):
            self.project_master_classes = list(model_classes)
            self.save_classes(self.project_master_classes)
            self.backup_project_classes()
            self.populate_classes(list(self.project_master_classes))
        self.model_class_names = model_classes
        dataset_lookup = {self.normalize_class_name(name): index for index, name in enumerate(self.canvas.classes)}
        mapping = {}
        for model_index, model_name in enumerate(model_classes):
            mapped_index = dataset_lookup.get(self.normalize_class_name(model_name))
            mapping[model_index] = mapped_index
        self.model_class_map = mapping
        matched = sum(1 for mapped_index in mapping.values() if mapped_index is not None)
        self.model_info.setText(
            f"Model: {os.path.basename(self.model_path)} | mapped {matched}/{len(model_classes)} classes"
        )
        self.refresh_suggest_class_filter()
        if matched < len(model_classes):
            missing = [model_classes[index] for index, mapped_index in mapping.items() if mapped_index is None]
            self.status.showMessage(
                f"Loaded model with partial class mapping. Unmapped model classes will be skipped: {', '.join(missing[:4])}",
                6000,
            )
        else:
            self.status.showMessage(f"Loaded model {os.path.basename(self.model_path)} with full class mapping", 4000)
        self.refresh_summary_labels()

    def normalize_class_name(self, name):
        normalized = "".join(char.lower() for char in str(name).strip() if char.isalnum())
        return CLASS_NAME_ALIASES.get(normalized, normalized)

    def merge_model_classes_into_project(self, model_classes, allowed_model_names=None):
        merged_classes = list(self.project_master_classes or self.canvas.classes)
        allowed_lookup = None
        if allowed_model_names:
            allowed_lookup = {self.normalize_class_name(name) for name in allowed_model_names}
        existing_lookup = {self.normalize_class_name(name): index for index, name in enumerate(merged_classes)}
        for model_name in model_classes:
            normalized_name = self.normalize_class_name(model_name)
            if allowed_lookup is not None and normalized_name not in allowed_lookup:
                continue
            if normalized_name not in existing_lookup:
                merged_classes.append(str(model_name).strip())
                existing_lookup[normalized_name] = len(merged_classes) - 1
        return merged_classes

    def ensure_model_classes_imported(self, allowed_model_names=None):
        if not self.detector or not self.model_class_names:
            return
        merged_classes = self.merge_model_classes_into_project(self.model_class_names, allowed_model_names)
        if merged_classes != self.canvas.classes:
            self.project_master_classes = list(merged_classes)
            self.save_classes(self.project_master_classes)
            self.backup_project_classes()
            self.populate_classes(merged_classes)
        else:
            self.build_model_class_mapping()

    def selected_model_class_names(self):
        if not (self.detector and self.model_class_names):
            return []
        selected_items = self.suggest_class_filter.selectedItems()
        if not selected_items:
            return []
        selected_names = {item.text() for item in selected_items}
        return [name for name in self.model_class_names if name in selected_names]

    def annotation_iou(self, left, right):
        a = left.normalized()
        b = right.normalized()
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union = a.width() * a.height() + b.width() * b.height() - intersection
        return intersection / union if union > 0 else 0.0

    def point_in_polygon(self, x, y, polygon_points):
        if len(polygon_points) < 3:
            return True
        inside = False
        j = len(polygon_points) - 1
        for i in range(len(polygon_points)):
            xi, yi = polygon_points[i]
            xj, yj = polygon_points[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def inside_roi(self, annotation, roi_points=None):
        if not roi_points:
            return True
        box = annotation.normalized()
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0
        return self.point_in_polygon(cx, cy, roi_points)

    def filter_new_detections(self, detections, existing_annotations, roi_points=None, allowed_class_ids=None):
        filtered = []
        iou_limit = self.iou_spin.value()
        for detection in detections:
            class_id, x1, y1, x2, y2 = detection[:5]
            score = detection[5] if len(detection) > 5 else -1.0
            if allowed_class_ids is not None and class_id not in allowed_class_ids:
                continue
            candidate = Annotation(x1, y1, x2, y2, class_id, score).normalized()
            if not self.inside_roi(candidate, roi_points):
                continue
            overlaps = False
            for existing in existing_annotations:
                if self.annotation_iou(candidate, existing) >= iou_limit:
                    overlaps = True
                    break
            if not overlaps:
                filtered.append(candidate)
        return filtered

    def read_annotations_for_image(self, image_name):
        size = self.get_image_size(image_name)
        if size is None:
            return []
        h, w = size
        label_path = self.label_path_for_image(image_name)
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id, xc, yc, bw, bh = map(float, parts)
                    except ValueError:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        continue
                    annotations.append(Annotation.from_yolo(class_id, xc, yc, bw, bh, w, h))
        return annotations

    def suggest_from_reference_image(self, reference_image_name, current_features=None, reference_features=None):
        current_image_path = self.current_image_path()
        if not current_image_path or not reference_image_name:
            return []
        reference_path = os.path.join(self.folder, reference_image_name)
        current_features = current_features or self.get_image_features(current_image_path)
        reference_features = reference_features or self.get_image_features(reference_path)
        if not current_features or not reference_features:
            return []

        ref_annotations = self.get_label_summary(reference_image_name)["annotations"]
        if not ref_annotations:
            return []

        ref_keypoints = reference_features["keypoints"]
        cur_keypoints = current_features["keypoints"]
        ref_descriptors = reference_features["descriptors"]
        cur_descriptors = current_features["descriptors"]
        if ref_descriptors is None or cur_descriptors is None or len(ref_keypoints) < 8 or len(cur_keypoints) < 8:
            return []

        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(ref_descriptors, cur_descriptors)
        except cv2.error:
            return []
        if len(matches) < 8:
            return []
        matches = sorted(matches, key=lambda match: match.distance)[:200]

        src = np.array([ref_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32)
        dst = np.array([cur_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32)
        try:
            homography, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        except cv2.error:
            return []
        if homography is None:
            return []

        height, width = current_features["gray"].shape[:2]
        suggestions = []
        for annotation in ref_annotations:
            box = annotation.normalized()
            corners = np.array(
                [[[box.x1, box.y1]], [[box.x2, box.y1]], [[box.x2, box.y2]], [[box.x1, box.y2]]],
                dtype="float32",
            )
            transformed = cv2.perspectiveTransform(corners, homography)
            xs = [point[0][0] for point in transformed]
            ys = [point[0][1] for point in transformed]
            mapped = Annotation(min(xs), min(ys), max(xs), max(ys), annotation.class_id, 0.5).normalized()
            mapped.clamp(width, height)
            if mapped.width() >= MIN_BOX_SIZE and mapped.height() >= MIN_BOX_SIZE:
                suggestions.append(mapped)
        return suggestions

    def count_reference_matches(self, reference_image_name, current_features=None, reference_features=None):
        current_image_path = self.current_image_path()
        if not current_image_path or not reference_image_name:
            return 0
        reference_path = os.path.join(self.folder, reference_image_name)
        current_features = current_features or self.get_image_features(current_image_path)
        reference_features = reference_features or self.get_image_features(reference_path)
        if not current_features or not reference_features:
            return 0
        ref_keypoints = reference_features["keypoints"]
        cur_keypoints = current_features["keypoints"]
        ref_descriptors = reference_features["descriptors"]
        cur_descriptors = current_features["descriptors"]
        if ref_descriptors is None or cur_descriptors is None or len(ref_keypoints) < 8 or len(cur_keypoints) < 8:
            return 0
        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(ref_descriptors, cur_descriptors)
        except cv2.error:
            return 0
        return len(matches)

    def write_annotations_for_image(self, image_name, annotations):
        size = self.get_image_size(image_name)
        if size is None:
            return
        label_path = self.label_path_for_image(image_name)
        if not annotations:
            if os.path.exists(label_path):
                try:
                    os.remove(label_path)
                except OSError:
                    return
            self.update_counts_for_label_state(image_name, False)
            self.invalidate_label_summary(image_name)
            return
        h, w = size
        lines = []
        for annotation in annotations:
            class_id, xc, yc, bw, bh = annotation.to_yolo(w, h)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        atomic_write_lines(label_path, lines)
        self.update_counts_for_label_state(image_name, len(annotations) > 0)
        self.invalidate_label_summary(image_name)

    def predict_boxes_for_image(self, image_path):
        if not self.detector:
            raise RuntimeError("No model loaded")
        results = self.detector.predict(
            source=image_path,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            verbose=False,
        )
        detections = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        cls_values = boxes.cls.cpu().numpy().astype(int)
        conf_values = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
        for index, (coords, class_id) in enumerate(zip(xyxy, cls_values)):
            mapped_class_id = self.model_class_map.get(int(class_id), int(class_id))
            if mapped_class_id is None or mapped_class_id >= len(self.canvas.classes):
                continue
            x1, y1, x2, y2 = [float(value) for value in coords[:4]]
            score = float(conf_values[index]) if conf_values is not None else -1.0
            detections.append((int(mapped_class_id), x1, y1, x2, y2, score))
        return detections

    def auto_annotate_current(self):
        image_path = self.current_image_path()
        if not image_path:
            return
        if not self.detector:
            self.load_model_file()
            if not self.detector:
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        selected_model_names = self.selected_model_class_names()
        if self.import_model_classes_check.isChecked():
            self.ensure_model_classes_imported(selected_model_names if selected_model_names else None)
        try:
            detections = self.predict_boxes_for_image(image_path)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Auto annotation failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        existing_suggestions = [] if self.replace_labels.isChecked() else list(self.canvas.suggestions)
        base_annotations = [] if self.replace_labels.isChecked() else list(self.canvas.annotations) + existing_suggestions
        roi_points = self.canvas.roi_points if self.canvas.has_roi() else None
        allowed_class_ids = self.selected_suggest_class_ids()
        suggestions = self.filter_new_detections(detections, base_annotations, roi_points, allowed_class_ids)
        self.replace_pending = self.replace_labels.isChecked()
        self.canvas.set_suggestions(existing_suggestions + suggestions)
        filter_text = "all classes" if allowed_class_ids is None else f"{len(allowed_class_ids)} selected classes"
        self.status.showMessage(
            f"Generated {len(suggestions)} suggestions for {os.path.basename(image_path)} using {filter_text}.",
            5000,
        )

    def maybe_auto_annotate_loaded_image(self):
        if self.auto_advance_mode == "none" or not self.current_image_path():
            return
        if not self.detector:
            self.status.showMessage("Auto-next is enabled, but no model is loaded.", 4000)
            return
        self.ensure_model_classes_imported()
        if self.auto_advance_mode == "roi" and not self.canvas.has_roi():
            self.status.showMessage("Auto-next ROI mode is enabled, but no ROI is available for this image.", 4000)
            return
        self.auto_annotate_current()

    def copy_previous_annotations(self):
        self.canvas.copy_previous_annotations(self.previous_annotations_cache)

    def copy_labels_to_next_suggestions(self):
        if not self.canvas.has_image():
            return
        if not self.canvas.annotations:
            self.status.showMessage("No labels on the current image to copy.", 3000)
            return
        next_index = self.current_image_index + 1
        if next_index >= len(self.image_files):
            self.status.showMessage("Already at the last image.", 3000)
            return
        next_name = self.image_files[next_index]
        if self.get_label_summary(next_name)["total"] > 0:
            self.status.showMessage("Next image already has labels. Copy skipped.", 4000)
            return
        self.pending_next_suggestions = [
            Annotation(item.x1, item.y1, item.x2, item.y2, item.class_id, item.score) for item in self.canvas.annotations
        ]
        self.pending_next_suggestion_image = next_name
        self.pending_next_suggestion_count = len(self.pending_next_suggestions)
        self.load_current_image(next_index)

    def undo_action(self):
        self.canvas.undo()

    def redo_action(self):
        self.canvas.redo()
        if self.autosave_on_image_change():
            self.save_current_labels()

    def reset_current_image_annotations(self):
        if self.canvas.has_roi():
            removed = self.canvas.delete_annotations_in_roi()
            if removed:
                self.status.showMessage(f"Deleted {removed} labels inside ROI.", 3000)
        else:
            self.canvas.reset_annotations()
        if self.autosave_on_image_change():
            self.save_current_labels()

    def reject_suggestions_only(self):
        if self.canvas.has_roi():
            removed = self.canvas.delete_suggestions_in_roi()
            if removed:
                self.status.showMessage(f"Rejected {removed} suggestions inside ROI.", 3000)
            return
        if not self.canvas.suggestions:
            return
        self.canvas.reject_all_suggestions()

    def accept_selected_suggestion(self):
        if self.canvas.has_roi():
            self.accept_suggestions_in_roi()
            return
        if self.replace_pending and self.canvas.suggestions:
            self.canvas.push_history()
            self.canvas.annotations = []
            self.replace_pending = False
        self.canvas.accept_selected_suggestion()
        if self.autosave_on_image_change():
            self.save_current_labels()

    def accept_all_suggestions(self):
        if self.canvas.has_roi():
            self.accept_suggestions_in_roi()
            return
        if self.replace_pending and self.canvas.suggestions:
            self.canvas.push_history()
            self.canvas.annotations = []
            self.replace_pending = False
        self.canvas.accept_all_suggestions()
        if self.autosave_on_image_change():
            self.save_current_labels()

    def accept_suggestions_in_roi(self):
        if not self.canvas.has_roi():
            self.status.showMessage("Draw an ROI first.", 3000)
            return
        if self.replace_pending and self.canvas.suggestions:
            self.canvas.push_history()
            self.canvas.annotations = []
            self.replace_pending = False
        accepted = self.canvas.accept_suggestions_in_roi()
        if accepted and self.autosave_on_image_change():
            self.save_current_labels()
        self.status.showMessage(f"Accepted {accepted} suggestions inside ROI.", 3000)

    def reject_all_suggestions(self):
        self.replace_pending = False
        self.canvas.reject_all_suggestions()

    def reject_selected_suggestion(self):
        if self.canvas.selected_suggestion_index < 0 and self.suggestion_list.currentRow() >= 0:
            self.canvas.selected_suggestion_index = self.suggestion_list.currentRow()
        if self.canvas.selected_suggestion_index < 0 and self.canvas.suggestions:
            self.canvas.selected_suggestion_index = 0
        self.canvas.reject_selected_suggestion()

    def clear_roi_selection(self):
        self.canvas.clear_roi()
        self.normalized_roi_points = []

    def delete_labels_in_roi(self):
        removed = self.canvas.delete_annotations_in_roi()
        if removed and self.autosave_on_image_change():
            self.save_current_labels()
        self.status.showMessage(f"Deleted {removed} labels inside ROI.", 3000)

    def set_labels_in_roi_to_current_class(self):
        if not self.canvas.has_roi():
            self.status.showMessage("Draw an ROI first.", 3000)
            return
        affected = self.canvas.collect_annotations_in_roi()
        if not affected:
            self.status.showMessage("No labels found inside ROI.", 3000)
            return
        self.canvas.push_history()
        for index in affected:
            self.canvas.annotations[index].class_id = self.canvas.current_class_id
        self.canvas.set_selected_indices(affected)
        self.canvas.annotations_changed.emit()
        self.canvas.update()
        if self.autosave_on_image_change():
            self.save_current_labels()
        self.status.showMessage(f"Changed {len(affected)} ROI labels to {self.canvas.class_name(self.canvas.current_class_id)}.", 3000)

    def set_selected_labels_to_current_class(self):
        if not self.canvas.selected_indices:
            self.status.showMessage("Select one or more labels first.", 3000)
            return
        self.canvas.push_history()
        for index in self.canvas.selected_indices:
            self.canvas.annotations[index].class_id = self.canvas.current_class_id
        self.canvas.annotations_changed.emit()
        self.canvas.update()
        if self.autosave_on_image_change():
            self.save_current_labels()
        self.status.showMessage(
            f"Changed {len(self.canvas.selected_indices)} selected labels to {self.canvas.class_name(self.canvas.current_class_id)}.",
            3000,
        )

    def toggle_reviewed_shortcut(self):
        self.reviewed_check.setChecked(not self.reviewed_check.isChecked())

    def toggle_boxes(self, checked):
        self.canvas.show_boxes = checked
        self.canvas.update()

    def focus_class_for_edit(self, class_id):
        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(0)
        self.shortcuts_suspended = True
        self.set_shortcuts_enabled(False)
        self.clear_class_type_buffer()
        self.class_picker.setCurrentIndex(class_id)
        row = self.class_list_row_by_id.get(class_id, class_id)
        if row >= 0:
            self.class_list.setCurrentRow(row)
            item = self.class_list.item(row)
            if item is not None:
                self.class_list.scrollToItem(item)
        self.class_list.setFocus()

    def focusInEvent(self, event):
        self.shortcuts_suspended = False
        self.set_shortcuts_enabled(True)
        super().focusInEvent(event)

    def navigate_from_canvas(self, direction):
        if direction < 0:
            self.previous_image()
        elif direction > 0:
            self.next_image()

    def toggle_roi_mode_shortcut(self):
        self.roi_mode_btn.setChecked(not self.roi_mode_btn.isChecked())

    def navigate_from_canvas(self, direction):
        if direction < 0:
            self.previous_image()
        elif direction > 0:
            self.next_image()

    def update_mouse_status(self, text):
        self.status.showMessage(text)

    def eventFilter(self, obj, event):
        if obj in (self.class_list, self.class_picker, self.class_list.viewport(), self.class_picker_view):
            if event.type() == QEvent.FocusIn:
                self.shortcuts_suspended = True
                self.set_shortcuts_enabled(False)
            elif event.type() == QEvent.FocusOut:
                if self.focusWidget() not in (self.class_list, self.class_picker):
                    self.shortcuts_suspended = False
                    self.clear_class_type_buffer()
                    self.set_shortcuts_enabled(True)
            elif event.type() == QEvent.InputMethod:
                return False
            elif event.type() == QEvent.KeyPress:
                if event.isAutoRepeat():
                    return False
                if event.key() == Qt.Key_Escape:
                    self.clear_class_type_buffer()
                    self.canvas.setFocus()
                    self.shortcuts_suspended = False
                    self.set_shortcuts_enabled(True)
                    return True
                if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    self.clear_class_type_buffer()
                    self.canvas.setFocus()
                    self.shortcuts_suspended = False
                    self.set_shortcuts_enabled(True)
                    return True
                if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right, Qt.Key_PageUp, Qt.Key_PageDown):
                    return False
                if event.key() == Qt.Key_Backspace:
                    if self.class_type_buffer:
                        self.class_type_buffer = self.class_type_buffer[:-1]
                        self.apply_class_type_buffer()
                    return True
                if event.text() == "":
                    return False
                if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
                    return False
                text = event.text()
                if text:
                    normalized = text.lower()
                    if len(normalized) == 1 and normalized.isalpha():
                        now = time.monotonic()
                        if (
                            normalized == self.class_type_last_key
                            and now - self.class_type_last_time < 1.0
                        ):
                            self.class_type_buffer = normalized
                        else:
                            if now - self.class_type_last_time > 1.2:
                                self.class_type_buffer = ""
                            self.class_type_buffer += normalized
                        self.class_type_last_key = normalized
                        self.class_type_last_time = now
                        self.class_type_timer.start(1200)
                        self.apply_class_type_buffer()
                        return True
                    return False
        return super().eventFilter(obj, event)

    def clear_class_type_buffer(self):
        self.class_type_buffer = ""
        self.class_type_last_key = ""
        self.class_type_cycle_index = -1
        self.class_type_cycle_query = ""

    def apply_class_type_buffer(self):
        query = self.class_type_buffer.strip().lower()
        if not query:
            return
        matches = []
        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if item is None:
                continue
            text = item.text()
            name = text.split("(", 1)[0].strip().lower()
            if name.startswith(query):
                matches.append((row, item.data(Qt.UserRole)))
        if not matches and len(query) > 1:
            self.class_type_buffer = query[-1]
            return self.apply_class_type_buffer()
        if not matches:
            return
        if len(query) == 1 and self.class_type_cycle_query == query:
            self.class_type_cycle_index = (self.class_type_cycle_index + 1) % len(matches)
        else:
            self.class_type_cycle_index = 0
        self.class_type_cycle_query = query
        row, match_id = matches[self.class_type_cycle_index]
        self.class_picker.setCurrentIndex(match_id)
        if row >= 0:
            self.class_list.setCurrentRow(row)
            item = self.class_list.item(row)
            if item is not None:
                self.class_list.scrollToItem(item)

    def next_image(self):
        self.sync_images_with_disk(reload_current=False)
        row = self.file_list.currentRow()
        if row < 0:
            row = self.current_image_index
        if row < self.file_list.count() - 1:
            next_row = row + 1
            self.file_list.setCurrentRow(next_row)
            if self.file_list.currentRow() != next_row:
                self.load_current_image(next_row)

    def previous_image(self):
        self.sync_images_with_disk(reload_current=False)
        row = self.file_list.currentRow()
        if row < 0:
            row = self.current_image_index
        if row > 0:
            prev_row = row - 1
            self.file_list.setCurrentRow(prev_row)
            if self.file_list.currentRow() != prev_row:
                self.load_current_image(prev_row)

    def next_unlabeled_image(self):
        self.sync_images_with_disk(reload_current=False)
        if not self.image_files:
            return
        start = max(self.current_image_index, -1) + 1
        for offset in range(len(self.image_files)):
            index = (start + offset) % len(self.image_files)
            if not self.has_saved_labels(self.image_files[index]):
                self.file_list.setCurrentRow(index)
                if self.file_list.currentRow() != index:
                    self.load_current_image(index)
                return
        self.status.showMessage("All images already have labels.", 3000)

    def select_class_by_index(self, index):
        if 0 <= index < len(self.canvas.classes):
            self.class_picker.setCurrentIndex(index)
            self.class_list.setCurrentRow(self.class_list_row_by_id.get(index, index))
            self.canvas.current_class_id = index

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_B and hasattr(self, "canvas"):
            self.canvas.force_new_box_mode = True
            self.canvas.setCursor(Qt.CrossCursor)
            self.canvas.setFocus(Qt.ActiveWindowFocusReason)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_B and hasattr(self, "canvas"):
            self.canvas.force_new_box_mode = False
            if self.canvas.has_image():
                pos = self.canvas.mapFromGlobal(QCursor.pos())
                image_x, image_y = self.canvas.display_to_image(pos.x(), pos.y())
                image_x, image_y = self.canvas.clamp_point(image_x, image_y)
                self.canvas.set_cursor_for_position(image_x, image_y)
            else:
                self.canvas.unsetCursor()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        answer = QMessageBox.question(
            self,
            "Close Annotator",
            "Close the annotator now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            event.ignore()
            return
        if self.autosave.isChecked() and self.save_current_labels() is False:
            event.ignore()
            return
        self.save_classes()
        self.save_settings()
        super().closeEvent(event)


def install_crash_logging():
    def excepthook(exc_type, exc_value, exc_traceback):
        crash_log_path = os.path.join(os.getcwd(), CRASH_LOG_NAME)
        with open(crash_log_path, "a", encoding="utf-8") as handle:
            handle.write("\n=== Crash ===\n")
            handle.write(f"Timestamp: {datetime.now().isoformat()}\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=handle)
        message = f"Unexpected error. Details were saved to {crash_log_path}"
        try:
            QMessageBox.critical(None, "Application Error", message)
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook


if __name__ == "__main__":
    install_crash_logging()
    app = QApplication(sys.argv)
    window = Annotator()
    window.show()
    sys.exit(app.exec_())
