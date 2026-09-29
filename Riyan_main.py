import cv2
import numpy as np
import math
import threading
import time
from queue import Queue, Empty

# =============================================================
# PLUGIN ARCHITECTURE
# These integrations are intentionally OPTIONAL. The current app
# runs with only OpenCV + NumPy. Each team member can implement
# one plugin without changing the core 3D math engine.
# =============================================================

# PLUGIN HERE ----------------------------------------------->
# REAL-TIME OPENCV VIDEO CAPTURE
# Team member: Computer Vision / Video
# Purpose: capture webcam frames continuously without blocking
# the graphics loop.
# To activate later, instantiate VideoCapturePlugin(0).
# ----------------------------------------------------------->

# PLUGIN HERE ----------------------------------------------->
# MULTITHREADING
# Team member: Systems / Integration
# Purpose: keep camera capture, computer vision and rendering
# on separate execution paths. The latest frame is shared
# through a thread-safe Queue.
# ----------------------------------------------------------->

# PLUGIN HERE ----------------------------------------------->
# COMPUTER VISION
# Team member: Computer Vision
# Purpose: process the live OpenCV frame. This is the extension
# point for face detection, object detection, optical flow,
# hand tracking, etc. without changing the 3D renderer.
# ----------------------------------------------------------->

# PLUGIN HERE ----------------------------------------------->
# PYGAME 3D GRAPHICS ENGINE
# Team member: Graphics Engine
# Purpose: an optional replacement/parallel renderer. Keep the
# current NumPy transformation pipeline and feed its projected
# vertices/faces into Pygame when this plugin is implemented.
# Pygame is NOT required to run the current OpenCV version.
# ----------------------------------------------------------->

ENABLE_VIDEO_PLUGIN = False
ENABLE_CV_PLUGIN = False
ENABLE_PYGAME_PLUGIN = False


class VideoCapturePlugin:
    """Threaded real-time OpenCV camera capture plugin."""

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.thread = None
        self.frames = Queue(maxsize=2)

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open OpenCV camera.")
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop, name="OpenCV-Camera", daemon=True
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            # Keep only the newest frame so vision never builds a backlog.
            while not self.frames.empty():
                try:
                    self.frames.get_nowait()
                except Empty:
                    break
            try:
                self.frames.put_nowait(frame)
            except Exception:
                pass

    def get_latest_frame(self):
        try:
            return self.frames.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ComputerVisionPlugin:
    """Safe extension point for real-time OpenCV vision."""

    def process(self, frame):
        # PLUGIN HERE --------------------------------------->
        # Add OpenCV computer-vision algorithms here.
        # Return: processed_frame, vision_data
        # Example future vision_data:
        # {"gesture": "left", "confidence": 0.91}
        # --------------------------------------------------->
        return frame, {}


class Pygame3DGraphicsPlugin:
    """Optional Pygame renderer interface.

    This deliberately does not import pygame. Install/import it
    only when the graphics team implements this plugin.
    """

    def render(self, vertices_3d, edges, faces=None):
        # PLUGIN HERE --------------------------------------->
        # Import pygame inside this method when this plugin is
        # implemented. Reuse the NumPy transformed vertices.
        # --------------------------------------------------->
        raise NotImplementedError(
            "Pygame plugin not enabled; use the OpenCV renderer."
        )

# ==============================================================
# LINEAR ALGEBRA 3D GRAPHICS VISUALIZER
# OpenCV + NumPy only
#
# Left  : interactive wireframe 3D object
# Right : live linear-algebra calculations
#
# Controls:
#   Move mouse over left panel -> X/Y rotation
#   Left/right mouse position   -> Y rotation
#   Up/down mouse position      -> X rotation
#   Keys 1-5                    -> select shape
#   O                           -> orthographic projection
#   P                           -> perspective projection
#   A                           -> toggle automatic Z rotation
#   R                           -> reset
#   ESC                         -> quit
# ==============================================================

W, H = 1400, 800
VIEW_W = 900
PANEL_W = W - VIEW_W

BG = (0, 0, 0)
PANEL_BG = (12, 12, 12)
GRID = (32, 32, 32)
WHITE = (235, 240, 248)
CYAN = (255, 205, 90)       # OpenCV uses BGR
GREEN = (100, 220, 130)
RED = (100, 110, 240)
YELLOW = (70, 210, 245)
MUTED = (150, 160, 175)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------
# Linear algebra
# --------------------------------------------------------------

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=float)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=float)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)


def scale_matrix(s):
    return np.diag([s, s, s]).astype(float)


def composite_matrix(ax, ay, az, scale=1.0):
    # Column-vector convention:
    # p' = M p
    # M = Rz * Ry * Rx * S
    S = scale_matrix(scale)
    return Rz(az) @ Ry(ay) @ Rx(ax) @ S


def apply_transform(vertices, M):
    # Each row is a point, so p' = M p becomes vertices @ M.T
    return vertices @ M.T


def orthographic_project(points, zoom=190):
    x = points[:, 0] * zoom + VIEW_W * 0.5
    y = -points[:, 1] * zoom + H * 0.5
    return np.column_stack((x, y))


def perspective_project(points, zoom=260, camera_distance=5.5):
    z = points[:, 2] + camera_distance
    z = np.maximum(z, 0.15)
    x = points[:, 0] * zoom / z + VIEW_W * 0.5
    y = -points[:, 1] * zoom / z + H * 0.5
    return np.column_stack((x, y))


# --------------------------------------------------------------
# Shape generation
# --------------------------------------------------------------

def cube():
    v = np.array([
        [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
        [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]
    ], dtype=float)
    e = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    return v, e


def cuboid():
    v = np.array([
        [-1.5,-1,-0.75], [1.5,-1,-0.75],
        [1.5,1,-0.75], [-1.5,1,-0.75],
        [-1.5,-1,0.75], [1.5,-1,0.75],
        [1.5,1,0.75], [-1.5,1,0.75]
    ], dtype=float)
    e = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    return v, e


def pyramid():
    v = np.array([
        [-1,-1,-1], [1,-1,-1], [1,-1,1], [-1,-1,1],
        [0,1.4,0]
    ], dtype=float)
    e = [
        (0,1),(1,2),(2,3),(3,0),
        (0,4),(1,4),(2,4),(3,4)
    ]
    return v, e


def cylinder(segments=28):
    v, e = [], []
    for y in (-1, 1):
        for i in range(segments):
            a = 2 * math.pi * i / segments
            v.append([math.cos(a), y, math.sin(a)])

    for i in range(segments):
        j = (i + 1) % segments
        e += [(i, j), (segments+i, segments+j), (i, segments+i)]

    return np.array(v, dtype=float), e


def hemisphere(segments=28, rings=8):
    # Upper hemisphere y >= 0
    v, e = [], []

    for r in range(rings + 1):
        phi = (math.pi / 2) * r / rings
        y = math.sin(phi)
        radius = math.cos(phi)

        for i in range(segments):
            a = 2 * math.pi * i / segments
            v.append([
                radius * math.cos(a),
                y,
                radius * math.sin(a)
            ])

    for r in range(rings + 1):
        for i in range(segments):
            j = (i + 1) % segments
            e.append((r * segments + i, r * segments + j))

    for r in range(rings):
        for i in range(segments):
            e.append((r * segments + i, (r + 1) * segments + i))

    return np.array(v, dtype=float), e


def shape_faces(name):
    # Faces are listed in a consistent winding order.
    if name in ("Cube", "Cuboid"):
        return [
            (0, 3, 2, 1),  # back
            (4, 5, 6, 7),  # front
            (0, 1, 5, 4),  # bottom
            (3, 7, 6, 2),  # top
            (0, 4, 7, 3),  # left
            (1, 2, 6, 5),  # right
        ]
    if name == "Pyramid":
        return [
            (0, 3, 2, 1),
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
        ]
    if name == "Cylinder":
        n = 28
        faces = [(tuple(range(n-1, -1, -1))), (tuple(range(n, 2*n)))]
        for i in range(n):
            j = (i + 1) % n
            faces.append((i, j, n+j, n+i))
        return faces
    if name == "Hemisphere":
        n, rings = 28, 8
        faces = []
        # Curved surface quads.
        for r in range(rings):
            for i in range(n):
                j = (i + 1) % n
                faces.append((r*n+i, r*n+j, (r+1)*n+j, (r+1)*n+i))
        # Flat circular base (equator).
        faces.append(tuple(range(n-1, -1, -1)))
        return faces
    return []

SHAPES = {
    "Cube": cube,
    "Cuboid": cuboid,
    "Hemisphere": hemisphere,
    "Cylinder": cylinder,
    "Pyramid": pyramid,
}


# --------------------------------------------------------------
# Drawing utilities
# --------------------------------------------------------------

def put(img, s, xy, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, str(s), xy, FONT, scale, color, thickness, cv2.LINE_AA)


def line(img, p1, p2, color, thickness=1):
    cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)),
             color, thickness, cv2.LINE_AA)


def rect(img, x1, y1, x2, y2, color, thickness=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def panel_title(img, title, y):
    put(img, title, (VIEW_W + 28, y), 0.58, CYAN, 2)


def matrix_lines(M, decimals=2):
    lines = []
    for row in M:
        lines.append(
            "[ " + "  ".join(f"{x: .{decimals}f}" for x in row) + " ]"
        )
    return lines


def draw_matrix(img, M, x, y, label, color=WHITE, decimals=2, gap=22):
    put(img, label, (x, y), 0.45, YELLOW, 1)
    yy = y + 23
    for s in matrix_lines(M, decimals):
        put(img, s, (x, yy), 0.39, color, 1)
        yy += gap
    return yy


def draw_grid(img):
    # viewport grid
    for x in range(0, VIEW_W, 50):
        cv2.line(img, (x, 0), (x, H), GRID, 1)
    for y in range(0, H, 50):
        cv2.line(img, (0, y), (VIEW_W, y), GRID, 1)

    cv2.line(img, (VIEW_W//2, 0), (VIEW_W//2, H), (55, 62, 77), 1)
    cv2.line(img, (0, H//2), (VIEW_W, H//2), (55, 62, 77), 1)


def draw_axes(img):
    origin = np.array([70, H-65])
    line(img, origin, origin + [90, 0], RED, 2)
    line(img, origin, origin + [0, -90], GREEN, 2)
    put(img, "X", (origin[0]+96, origin[1]+5), 0.45, RED, 1)
    put(img, "Y", (origin[0]-7, origin[1]-98), 0.45, GREEN, 1)


# --------------------------------------------------------------
# Main application
# --------------------------------------------------------------

state = {
    "mouse_x": VIEW_W // 2,
    "mouse_y": H // 2,
    "shape": "Cube",
    "projection": "Orthographic",
    "auto_z": True,
    "reset": False,
    "smooth_ax": 0.0,
    "smooth_ay": 0.0,
    "hold": False,
    "held_az": 0.0,
}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        state["mouse_x"] = x
        state["mouse_y"] = y


def main():
    # ========================================================
    # OPTIONAL PLUGINS
    # All plugin variables are initialized here so the core
    # application can run safely even when no plugins are enabled.
    # ========================================================
    video_plugin = None
    cv_plugin = None
    pygame_plugin = None

    # PLUGIN HERE ------------------------------------------->
    # REAL-TIME OPENCV VIDEO CAPTURE
    # Set ENABLE_VIDEO_PLUGIN = True when the camera teammate
    # is ready to integrate the threaded camera capture.
    # ------------------------------------------------------->
    if ENABLE_VIDEO_PLUGIN:
        video_plugin = VideoCapturePlugin(camera_index=0)
        video_plugin.start()

    # PLUGIN HERE ------------------------------------------->
    # COMPUTER VISION
    # ------------------------------------------------------->
    if ENABLE_CV_PLUGIN:
        cv_plugin = ComputerVisionPlugin()

    # PLUGIN HERE ------------------------------------------->
    # PYGAME 3D GRAPHICS ENGINE
    # ------------------------------------------------------->
    if ENABLE_PYGAME_PLUGIN:
        pygame_plugin = Pygame3DGraphicsPlugin()

    cv2.namedWindow("Linear Algebra 3D Visualizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Linear Algebra 3D Visualizer", W, H)
    cv2.setMouseCallback("Linear Algebra 3D Visualizer", mouse_callback)

    start = cv2.getTickCount() / cv2.getTickFrequency()

    shape_vertices, edges = SHAPES[state["shape"]]()
    faces = shape_faces(state["shape"])

    while True:
        now = cv2.getTickCount() / cv2.getTickFrequency()
        t = now - start

        # ---------------- Mouse -> rotation ----------------
        # Mouse only controls rotation when it is over the left viewport.
        mx = np.clip(state["mouse_x"], 0, VIEW_W)
        my = np.clip(state["mouse_y"], 0, H)

        rel_x = (mx - VIEW_W/2) / (VIEW_W/2)
        rel_y = (my - H/2) / (H/2)

        target_ax = rel_y * math.radians(75)
        target_ay = rel_x * math.radians(75)

        # Smooth the mouse response. When HOLD is enabled, keep the
        # current orientation exactly where it is.
        if not state["hold"]:
            state["smooth_ax"] += (target_ax - state["smooth_ax"]) * 0.055
            state["smooth_ay"] += (target_ay - state["smooth_ay"]) * 0.055
            state["held_az"] = t * 0.22 if state["auto_z"] else state["held_az"]

        ax = state["smooth_ax"]
        ay = state["smooth_ay"]
        az = state["held_az"]

        # ---------------- Transform ----------------
        M = composite_matrix(ax, ay, az)
        transformed = apply_transform(shape_vertices, M)

        # ---------------- Projection ----------------
        if state["projection"] == "Orthographic":
            projected = orthographic_project(transformed)
        else:
            projected = perspective_project(transformed)

        # ---------------- Canvas ----------------
        img = np.full((H, W, 3), BG, dtype=np.uint8)

        # PLUGIN HERE ------------------------------------------->
        # REAL-TIME VIDEO STREAM + COMPUTER VISION
        # Get the newest camera frame without blocking the renderer.
        # The future CV plugin can use the frame to control the
        # transformation angles or provide an overlay.
        # ------------------------------------------------------->
        latest_frame = None
        vision_data = {}
        if video_plugin is not None:
            latest_frame = video_plugin.get_latest_frame()
            if latest_frame is not None and cv_plugin is not None:
                latest_frame, vision_data = cv_plugin.process(latest_frame)

        draw_grid(img)
        cv2.rectangle(img, (VIEW_W, 0), (W, H), PANEL_BG, -1)
        cv2.line(img, (VIEW_W, 0), (VIEW_W, H), (75, 82, 98), 2)

        # ---------------- 3D object ----------------
        # Draw opaque faces first.  Faces are depth-sorted using their
        # average transformed Z so nearer surfaces appear on top.
        face_items = []
        for face in faces:
            pts3 = transformed[list(face)]
            avg_z = float(np.mean(pts3[:, 2]))
            pts2 = projected[list(face)].astype(np.int32)
            if np.all(np.isfinite(pts2)):
                face_items.append((avg_z, pts2))

        for _, pts2 in sorted(face_items, key=lambda item: item[0]):
            cv2.fillConvexPoly(img, pts2, (42, 115, 170), lineType=cv2.LINE_AA)
            cv2.polylines(img, [pts2], True, CYAN, 2, cv2.LINE_AA)

        # For curved shapes, use depth-shaded wire surfaces so they read
        # as solid/opaque while retaining their mathematical construction.
        if state["shape"] in ("Cylinder", "Hemisphere"):
            for a, b in edges:
                p1, p2 = projected[a], projected[b]
                if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
                    za = transformed[a, 2]
                    zb = transformed[b, 2]
                    avgz = (za + zb) / 2
                    intensity = int(np.clip(120 + avgz * 35, 65, 190))
                    c = (intensity, min(210, intensity + 30), min(255, intensity + 65))
                    line(img, p1, p2, c, 1)

        for a, b in edges:
            p1 = projected[a]
            p2 = projected[b]

            # Don't draw wildly invalid projected points.
            if np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)):
                line(img, p1, p2, CYAN, 2)

        for p in projected:
            if np.all(np.isfinite(p)):
                x, y = map(int, p)
                if 0 <= x < VIEW_W and 0 <= y < H:
                    cv2.circle(img, (x, y), 4, WHITE, -1, cv2.LINE_AA)

        draw_axes(img)

        put(img, state["shape"], (25, 38), 0.75, WHITE, 2)
        put(img, "Move mouse over the LEFT side to rotate",
            (25, H-20), 0.48, MUTED, 1)

        # ---------------- Right math panel ----------------
        x = VIEW_W + 28

        panel_title(img, "LIVE LINEAR ALGEBRA", 35)
        put(img, "Mouse position", (x, 65), 0.42, MUTED, 1)
        put(img, f"x = {int(mx):4d}    y = {int(my):4d}",
            (x, 87), 0.43, WHITE, 1)

        put(img, "Rotation angles", (x, 118), 0.42, MUTED, 1)
        put(img, f"θx = {math.degrees(ax):6.2f}°",
            (x, 140), 0.43, WHITE, 1)
        put(img, f"θy = {math.degrees(ay):6.2f}°",
            (x, 162), 0.43, WHITE, 1)
        put(img, f"θz = {math.degrees(az)%360:6.2f}°",
            (x, 184), 0.43, WHITE, 1)

        # ----------------------------------------------------------
        # Compact live matrix panel.  Keeping the three rotation
        # matrices side-by-side prevents the orthogonality and
        # projection sections from overlapping.
        # ----------------------------------------------------------
        matrix_y = 218
        put(img, "ROTATION MATRICES", (x, matrix_y), 0.42, YELLOW, 1)

        col1 = x
        col2 = x + 145
        col3 = x + 290

        def compact_matrix(Mx, cx, cy, label):
            put(img, label, (cx, cy), 0.38, MUTED, 1)
            yy = cy + 20
            for row in Mx:
                put(img, "[" + " ".join(f"{v: .2f}" for v in row) + "]",
                    (cx, yy), 0.32, WHITE, 1)
                yy += 18

        compact_matrix(Rx(ax), col1, matrix_y + 25, "Rx")
        compact_matrix(Ry(ay), col2, matrix_y + 25, "Ry")
        compact_matrix(Rz(az), col3, matrix_y + 25, "Rz")

        composite_y = 325
        put(img, "COMPOSITE TRANSFORMATION", (x, composite_y),
            0.42, YELLOW, 1)
        put(img, "M = Rz · Ry · Rx", (x, composite_y + 24),
            0.40, WHITE, 1)
        yy = composite_y + 48
        for row in M:
            put(img, "[ " + "  ".join(f"{v: .2f}" for v in row) + " ]",
                (x, yy), 0.36, WHITE, 1)
            yy += 19

        # Orthogonality check for the pure rotation matrix.
        R = Rz(az) @ Ry(ay) @ Rx(ax)
        RtR = R.T @ R
        orthogonal = np.allclose(RtR, np.eye(3), atol=1e-5)

        ortho_y = 430
        panel_title(img, "ORTHOGONALITY CHECK", ortho_y)
        put(img, "RᵀR =", (x, ortho_y + 27), 0.40, YELLOW, 1)
        yy = ortho_y + 50
        for row in RtR:
            put(img, "[ " + "  ".join(f"{v:.2f}" for v in row) + " ]",
                (x, yy), 0.36, WHITE, 1)
            yy += 19
        put(img, "✓ R is orthogonal" if orthogonal else "✗ Not orthogonal",
            (x, yy + 4), 0.40, GREEN if orthogonal else RED, 1)

        # Projection section now has its own fixed area and cannot
        # overlap the orthogonality result.
        proj_y = 575
        panel_title(img, "PROJECTION", proj_y)
        if state["projection"] == "Orthographic":
            put(img, "(x, y, z)  →  (x, y)", (x, proj_y + 28),
                0.40, WHITE, 1)
            put(img, "Depth does not change scale", (x, proj_y + 52),
                0.37, MUTED, 1)
        else:
            put(img, "x' = x·f / (z+d)", (x, proj_y + 28),
                0.38, WHITE, 1)
            put(img, "y' = y·f / (z+d)", (x, proj_y + 51),
                0.38, WHITE, 1)

        # Current interaction state.
        state_color = YELLOW if state["hold"] else GREEN
        put(img, "HOLD: ON — shape is still" if state["hold"] else "HOLD: OFF — mouse controls rotation",
            (x, 675), 0.36, state_color, 1)
        put(img, "Press H to hold/release", (x, 698), 0.36, MUTED, 1)

        # Bottom controls.
        put(img, "1 Cube   2 Cuboid   3 Hemisphere",
            (25, 65), 0.42, MUTED, 1)
        put(img, "4 Cylinder   5 Pyramid",
            (25, 88), 0.42, MUTED, 1)
        put(img, "O Orthographic   P Perspective",
            (25, 111), 0.42, MUTED, 1)
        put(img, "A Auto-Z   H Hold   R Reset   ESC Quit",
            (25, 134), 0.42, MUTED, 1)

        # ---------------- Keyboard ----------------
        key = cv2.waitKey(16) & 0xFF

        if key == 27:
            break

        if key == ord('1'):
            state["shape"] = "Cube"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key == ord('2'):
            state["shape"] = "Cuboid"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key == ord('3'):
            state["shape"] = "Hemisphere"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key == ord('4'):
            state["shape"] = "Cylinder"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key == ord('5'):
            state["shape"] = "Pyramid"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key in (ord('o'), ord('O')):
            state["projection"] = "Orthographic"
        elif key in (ord('p'), ord('P')):
            state["projection"] = "Perspective"
        elif key in (ord('a'), ord('A')):
            state["auto_z"] = not state["auto_z"]
        elif key in (ord('h'), ord('H')):
            state["hold"] = not state["hold"]
            if state["hold"]:
                state["held_az"] = az
        elif key in (ord('r'), ord('R')):
            state["mouse_x"] = VIEW_W // 2
            state["mouse_y"] = H // 2
            state["smooth_ax"] = 0.0
            state["smooth_ay"] = 0.0
            start = cv2.getTickCount() / cv2.getTickFrequency()

        cv2.imshow("Linear Algebra 3D Visualizer", img)

    # PLUGIN HERE ----------------------------------------------->
    # Stop the background camera thread cleanly.
    # ----------------------------------------------------------->
    if video_plugin is not None:
        video_plugin.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
