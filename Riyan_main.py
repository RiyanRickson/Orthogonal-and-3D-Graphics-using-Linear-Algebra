import cv2
import numpy as np
import math
import threading
import time
from queue import Queue, Empty

# =============================================================
# PLUGIN ARCHITECTURE
# These integrations are intentionally OPTIONAL.
# Current core: OpenCV + NumPy.
# =============================================================

# PLUGIN HERE ----------------------------------------------->
# REAL-TIME OPENCV VIDEO CAPTURE
# Team member 4: threaded webcam/network stream.
# ------------------------------------------------------------>

# PLUGIN HERE ----------------------------------------------->
# MULTITHREADING
# Team member 4: background frame acquisition / processing.
# ------------------------------------------------------------>

# PLUGIN HERE ----------------------------------------------->
# COMPUTER VISION
# Team member 4: feature/gesture/object detection.
# ------------------------------------------------------------>

# PLUGIN HERE ----------------------------------------------->
# PYGAME 3D GRAPHICS ENGINE
# Team member 3/4: optional alternate renderer using the same
# NumPy transformation pipeline.
# ------------------------------------------------------------>

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
            target=self._capture_loop,
            name="OpenCV-Camera",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
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
    """Extension point for real-time OpenCV computer vision."""
    def process(self, frame):
        # PLUGIN HERE --------------------------------------->
        # Add face/object/gesture/feature detection here.
        # Return: processed_frame, vision_data
        # --------------------------------------------------->
        return frame, {}


class Pygame3DGraphicsPlugin:
    """Optional Pygame renderer interface."""
    def render(self, vertices_3d, edges, faces=None):
        # PLUGIN HERE --------------------------------------->
        # Import pygame here when this plugin is implemented.
        # Reuse the NumPy transformed/projected vertices.
        # --------------------------------------------------->
        raise NotImplementedError("Pygame renderer is not enabled.")


# =============================================================
# Configuration
# =============================================================
W, H = 1400, 800
VIEW_W = 900
PANEL_W = W - VIEW_W

BG = (0, 0, 0)
PANEL_BG = (10, 10, 10)
GRID = (30, 30, 30)
WHITE = (235, 240, 248)
CYAN = (255, 205, 90)      # OpenCV uses BGR
GREEN = (100, 220, 130)
RED = (100, 110, 240)
YELLOW = (70, 210, 245)
ORANGE = (60, 160, 255)
MUTED = (145, 155, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================
# Linear algebra
# =============================================================

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=float)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=float)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=float)


def to4(A3):
    """Embed a 3x3 linear transformation into homogeneous 4x4 form."""
    A4 = np.eye(4, dtype=float)
    A4[:3, :3] = A3
    return A4


def shear_matrix(shxy=0.0, shxz=0.0, shyx=0.0,
                 shyz=0.0, shzx=0.0, shzy=0.0):
    """Homogeneous 4x4 shear matrix."""
    return np.array([
        [1,    shxy, shxz, 0],
        [shyx, 1,    shyz, 0],
        [shzx, shzy, 1,    0],
        [0,    0,    0,    1],
    ], dtype=float)


def scale_matrix4(s):
    return np.diag([s, s, s, 1.0]).astype(float)


def composite_matrix(ax, ay, az, scale=1.0, shear_on=False):
    """Homogeneous composite transformation.

    p' = M p
    M = Rz * Ry * Rx * Shear * Scale
    """
    S = scale_matrix4(scale)
    H = shear_matrix(
        shxy=0.18 if shear_on else 0.0,
        shxz=0.10 if shear_on else 0.0,
        shyz=0.12 if shear_on else 0.0,
    )
    return to4(Rz(az)) @ to4(Ry(ay)) @ to4(Rx(ax)) @ H @ S


def apply_transform(vertices, M4):
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    transformed_h = homogeneous @ M4.T
    w = np.where(np.abs(transformed_h[:, 3]) < 1e-9, 1.0,
                 transformed_h[:, 3])
    return transformed_h[:, :3] / w[:, None]


def orthographic_projection_matrix(left=-3.8, right=3.8,
                                   bottom=-3.0, top=3.0,
                                   near=-10.0, far=10.0):
    """Conventional 4x4 orthographic projection matrix."""
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1],
    ], dtype=float)


def perspective_projection_matrix(fov_deg=60.0, aspect=VIEW_W/H,
                                   near=0.1, far=100.0):
    """Conventional 4x4 perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0],
    ], dtype=float)


def project_points(points, projection="Orthographic"):
    """Apply a real 4x4 projection matrix and map NDC to pixels."""
    P = (orthographic_projection_matrix()
         if projection == "Orthographic"
         else perspective_projection_matrix())

    h = np.column_stack((points, np.ones(len(points))))
    clip = h @ P.T
    w = np.where(np.abs(clip[:, 3]) < 1e-9, 1.0, clip[:, 3])
    ndc = clip[:, :3] / w[:, None]

    x = (ndc[:, 0] + 1.0) * 0.5 * VIEW_W
    y = (1.0 - ndc[:, 1]) * 0.5 * H
    return np.column_stack((x, y)), P, ndc


def transformation_order_matrices(ax, ay, az):
    """Two intentionally different valid orders for demonstration."""
    rx = to4(Rx(ax))
    ry = to4(Ry(ay))
    rz = to4(Rz(az))
    order_a = rz @ ry @ rx
    order_b = rx @ ry @ rz
    return order_a, order_b


# =============================================================
# Shape generation
# =============================================================

def cube():
    v = np.array([
        [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
        [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1]
    ], dtype=float)
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
    return v, e


def cuboid():
    v = np.array([
        [-1.5,-1,-0.75], [1.5,-1,-0.75], [1.5,1,-0.75], [-1.5,1,-0.75],
        [-1.5,-1,0.75], [1.5,-1,0.75], [1.5,1,0.75], [-1.5,1,0.75]
    ], dtype=float)
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
    return v, e


def pyramid():
    v = np.array([
        [-1,-1,-1], [1,-1,-1], [1,-1,1], [-1,-1,1], [0,1.4,0]
    ], dtype=float)
    e = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]
    return v, e


def cylinder(segments=28):
    v, e = [], []
    for y in (-1, 1):
        for i in range(segments):
            a = 2*math.pi*i/segments
            v.append([math.cos(a), y, math.sin(a)])
    for i in range(segments):
        j = (i+1) % segments
        e += [(i,j), (segments+i,segments+j), (i,segments+i)]
    return np.array(v, dtype=float), e


def hemisphere(segments=28, rings=8):
    v, e = [], []
    for r in range(rings+1):
        phi = (math.pi/2)*r/rings
        y = math.sin(phi)
        radius = math.cos(phi)
        for i in range(segments):
            a = 2*math.pi*i/segments
            v.append([radius*math.cos(a), y, radius*math.sin(a)])
    for r in range(rings+1):
        for i in range(segments):
            j = (i+1) % segments
            e.append((r*segments+i, r*segments+j))
    for r in range(rings):
        for i in range(segments):
            e.append((r*segments+i, (r+1)*segments+i))
    return np.array(v, dtype=float), e


def shape_faces(name):
    if name in ("Cube", "Cuboid"):
        return [(0,3,2,1),(4,5,6,7),(0,1,5,4),
                (3,7,6,2),(0,4,7,3),(1,2,6,5)]
    if name == "Pyramid":
        return [(0,3,2,1),(0,1,4),(1,2,4),(2,3,4),(3,0,4)]
    if name == "Cylinder":
        n = 28
        faces = [tuple(range(n-1,-1,-1)), tuple(range(n,2*n))]
        for i in range(n):
            j = (i+1) % n
            faces.append((i,j,n+j,n+i))
        return faces
    if name == "Hemisphere":
        n, rings = 28, 8
        faces = []
        for r in range(rings):
            for i in range(n):
                j = (i+1) % n
                faces.append((r*n+i,r*n+j,(r+1)*n+j,(r+1)*n+i))
        faces.append(tuple(range(n-1,-1,-1)))
        return faces
    return []


SHAPES = {"Cube": cube, "Cuboid": cuboid, "Hemisphere": hemisphere,
          "Cylinder": cylinder, "Pyramid": pyramid}


# =============================================================
# UI helpers
# =============================================================

def put(img, s, xy, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, str(s), xy, FONT, scale, color, thickness, cv2.LINE_AA)


def line(img, p1, p2, color, thickness=1):
    cv2.line(img, tuple(map(int,p1)), tuple(map(int,p2)), color, thickness, cv2.LINE_AA)


def panel_title(img, title, y):
    put(img, title, (VIEW_W+28,y), 0.54, CYAN, 2)


def draw_matrix(img, M, x, y, label, scale=0.32, gap=18, decimals=2):
    put(img, label, (x,y), 0.38, YELLOW, 1)
    yy = y+20
    for row in M:
        put(img, "[ " + " ".join(f"{v: .{decimals}f}" for v in row) + " ]",
            (x,yy), scale, WHITE, 1)
        yy += gap
    return yy


def draw_grid(img):
    for x in range(0,VIEW_W,50):
        cv2.line(img,(x,0),(x,H),GRID,1)
    for y in range(0,H,50):
        cv2.line(img,(0,y),(VIEW_W,y),GRID,1)
    cv2.line(img,(VIEW_W//2,0),(VIEW_W//2,H),(50,50,50),1)
    cv2.line(img,(0,H//2),(VIEW_W,H//2),(50,50,50),1)


def draw_axes(img):
    o=np.array([70,H-65])
    line(img,o,o+[90,0],RED,2)
    line(img,o,o+[0,-90],GREEN,2)
    put(img,"X",(o[0]+96,o[1]+5),0.45,RED,1)
    put(img,"Y",(o[0]-7,o[1]-98),0.45,GREEN,1)


# =============================================================
# State + mouse
# =============================================================
state = {
    "mouse_x": VIEW_W//2,
    "mouse_y": H//2,
    "shape": "Cube",
    "projection": "Orthographic",
    "auto_z": True,
    "hold": False,
    "shear": False,
    "order_demo": False,
    "smooth_ax": 0.0,
    "smooth_ay": 0.0,
    "held_az": 0.0,
}


def mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        state["mouse_x"], state["mouse_y"] = x,y


# =============================================================
# Main
# =============================================================

def main():
    video_plugin = None
    cv_plugin = None
    pygame_plugin = None

    # PLUGIN HERE ----------------------------------------------->
    # REAL-TIME OPENCV VIDEO CAPTURE
    # ------------------------------------------------------------>
    if ENABLE_VIDEO_PLUGIN:
        video_plugin = VideoCapturePlugin(0)
        video_plugin.start()

    # PLUGIN HERE ----------------------------------------------->
    # COMPUTER VISION
    # ------------------------------------------------------------>
    if ENABLE_CV_PLUGIN:
        cv_plugin = ComputerVisionPlugin()

    # PLUGIN HERE ----------------------------------------------->
    # PYGAME 3D GRAPHICS ENGINE
    # ------------------------------------------------------------>
    if ENABLE_PYGAME_PLUGIN:
        pygame_plugin = Pygame3DGraphicsPlugin()

    cv2.namedWindow("Linear Algebra 3D Visualizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Linear Algebra 3D Visualizer", W,H)
    cv2.setMouseCallback("Linear Algebra 3D Visualizer", mouse_callback)

    start = time.perf_counter()
    shape_vertices, edges = SHAPES[state["shape"]]()
    faces = shape_faces(state["shape"])

    while True:
        t = time.perf_counter() - start
        mx = float(np.clip(state["mouse_x"],0,VIEW_W))
        my = float(np.clip(state["mouse_y"],0,H))
        rel_x = (mx-VIEW_W/2)/(VIEW_W/2)
        rel_y = (my-H/2)/(H/2)
        target_ax = rel_y*math.radians(75)
        target_ay = rel_x*math.radians(75)

        if not state["hold"]:
            state["smooth_ax"] += (target_ax-state["smooth_ax"])*0.045
            state["smooth_ay"] += (target_ay-state["smooth_ay"])*0.045
            if state["auto_z"]:
                state["held_az"] = t*0.22

        ax, ay, az = state["smooth_ax"],state["smooth_ay"],state["held_az"]
        M = composite_matrix(ax,ay,az,shear_on=state["shear"])
        transformed = apply_transform(shape_vertices,M)
        projected, P, ndc = project_points(transformed,state["projection"])

        img=np.full((H,W,3),BG,dtype=np.uint8)
        draw_grid(img)
        cv2.rectangle(img,(VIEW_W,0),(W,H),PANEL_BG,-1)
        cv2.line(img,(VIEW_W,0),(VIEW_W,H),(65,65,65),2)

        # PLUGIN HERE ------------------------------------------->
        # REAL-TIME VIDEO STREAM + COMPUTER VISION
        # Fetch newest frame without blocking the 3D renderer.
        # ------------------------------------------------------->
        if video_plugin is not None:
            frame=video_plugin.get_latest_frame()
            if frame is not None and cv_plugin is not None:
                frame,vision_data=cv_plugin.process(frame)

        # ------------------------------------------------------
        # EXPLICIT TRANSFORMATION-ORDER DEMO
        # The normal object uses Rz·Ry·Rx. When M is enabled, an
        # orange ghost uses Rx·Ry·Rz so the non-commutative effect
        # is visible directly on the screen.
        # ------------------------------------------------------
        if state["order_demo"]:
            _, reversed_order = transformation_order_matrices(ax, ay, az)
            reversed_vertices = apply_transform(shape_vertices, reversed_order)
            reversed_projected, _, _ = project_points(
                reversed_vertices, state["projection"]
            )
            for a, b in edges:
                if (np.all(np.isfinite(reversed_projected[a])) and
                        np.all(np.isfinite(reversed_projected[b]))):
                    line(img, reversed_projected[a], reversed_projected[b], ORANGE, 1)
            put(img, "ORANGE = Rx·Ry·Rz", (20, 138), 0.35, ORANGE, 1)

        # Opaque faces, depth sorted.
        items=[]
        for face in faces:
            pts3=transformed[list(face)]
            pts2=projected[list(face)].astype(np.int32)
            if np.all(np.isfinite(pts2)):
                items.append((float(np.mean(pts3[:,2])),pts2))
        for _,pts2 in sorted(items,key=lambda z:z[0]):
            cv2.fillConvexPoly(img,pts2,(35,95,145),lineType=cv2.LINE_AA)
            cv2.polylines(img,[pts2],True,CYAN,2,cv2.LINE_AA)

        for a,b in edges:
            if np.all(np.isfinite(projected[a])) and np.all(np.isfinite(projected[b])):
                line(img,projected[a],projected[b],CYAN,2)
        for p in projected:
            x,y=map(int,p)
            if 0<=x<VIEW_W and 0<=y<H:
                cv2.circle(img,(x,y),4,WHITE,-1,cv2.LINE_AA)
        draw_axes(img)
        put(img,state["shape"],(25,38),0.75,WHITE,2)

        # ------------------------------------------------------
        # RIGHT PANEL
        # ------------------------------------------------------
        x=VIEW_W+28
        panel_title(img,"LIVE LINEAR ALGEBRA",35)
        put(img,f"Mouse: x={int(mx)}  y={int(my)}",(x,67),0.39,WHITE,1)
        put(img,f"θx={math.degrees(ax):5.1f}°  θy={math.degrees(ay):5.1f}°  θz={math.degrees(az)%360:5.1f}°",
            (x,89),0.37,MUTED,1)

        put(img,"ROTATION MATRICES",(x,119),0.40,YELLOW,1)
        for cx,label,mat in [(x,"Rx",Rx(ax)),(x+145,"Ry",Ry(ay)),(x+290,"Rz",Rz(az))]:
            put(img,label,(cx,142),0.37,MUTED,1)
            yy=162
            for row in mat:
                put(img,"["+" ".join(f"{v:.2f}" for v in row)+"]",(cx,yy),0.30,WHITE,1)
                yy+=17

        comp_y=218
        put(img,"COMPOSITE TRANSFORMATION",(x,comp_y),0.40,YELLOW,1)
        put(img,"M = Rz · Ry · Rx · H · S",(x,241),0.37,WHITE,1)
        yy=262
        for row in M:
            put(img,"["+" ".join(f"{v:.2f}" for v in row)+"]",(x,yy),0.29,WHITE,1)
            yy+=17

        # Explicit order demonstration.
        order_y=345
        put(img,"WHY ORDER MATTERS",(x,order_y),0.40,ORANGE,1)
        A,B=transformation_order_matrices(ax,ay,az)
        put(img,"A = Rz · Ry · Rx",(x,368),0.35,WHITE,1)
        put(img,"B = Rx · Ry · Rz",(x,388),0.35,WHITE,1)
        diff=np.linalg.norm(A-B)
        put(img,f"||A − B|| = {diff:.3f}",(x,408),0.35,
            GREEN if diff<1e-6 else YELLOW,1)
        put(img,"Different matrices → different result" if diff>=1e-6 else "Same result for this angle",
            (x,428),0.31,MUTED,1)

        # Orthogonality is checked on R only, not on shear/composite M.
        R=Rz(az)@Ry(ay)@Rx(ax)
        RtR=R.T@R
        orth=np.allclose(RtR,np.eye(3),atol=1e-5)
        ortho_y=455
        panel_title(img,"ORTHOGONALITY CHECK",ortho_y)
        put(img,"RᵀR =",(x,480),0.38,YELLOW,1)
        yy=501
        for row in RtR:
            put(img,"["+" ".join(f"{v:.2f}" for v in row)+"]",(x,yy),0.31,WHITE,1)
            yy+=17
        put(img,"✓ R is orthogonal",(x,554),0.35,GREEN if orth else RED,1)

        # Shear section.
        shear_y=582
        put(img,f"SHEAR MATRIX  [{ 'ON' if state['shear'] else 'OFF' }]",(x,shear_y),0.40,ORANGE,1)
        Hm=shear_matrix(0.18 if state["shear"] else 0.0,
                        0.10 if state["shear"] else 0.0,
                        0.0,0.12 if state["shear"] else 0.0,0.0,0.0)
        yy=shear_y+22
        for row in Hm:
            put(img,"["+" ".join(f"{v:.2f}" for v in row)+"]",(x,yy),0.28,WHITE,1)
            yy+=16

        # Projection matrix: actual conventional 4x4 matrix used by the renderer.
        proj_y=680
        put(img,f"4×4 {state['projection'].upper()} PROJECTION",(x,proj_y),0.38,YELLOW,1)
        yy=702
        for row in P:
            put(img,"["+" ".join(f"{v:.2f}" for v in row)+"]",(x,yy),0.27,WHITE,1)
            yy+=16

        # Controls at bottom-left.
        put(img,"1 Cube  2 Cuboid  3 Hemisphere  4 Cylinder  5 Pyramid",(20,70),0.37,MUTED,1)
        put(img,"O Orthographic   P Perspective   S Shear",(20,92),0.37,MUTED,1)
        put(img,"M Order Demo   A Auto-Z   H Hold   R Reset   ESC Quit",(20,114),0.37,MUTED,1)
        put(img,"Move mouse over LEFT viewport to rotate",(20,H-18),0.43,MUTED,1)

        # ------------------------------------------------------
        # Keyboard
        # ------------------------------------------------------
        key=cv2.waitKey(16)&0xFF
        if key==27:
            break
        # Shape selection must reload vertices, edges and faces.
        # Otherwise only the label changes and the old cube geometry remains.
        if key==ord('1'):
            state["shape"]="Cube"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key==ord('2'):
            state["shape"]="Cuboid"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key==ord('3'):
            state["shape"]="Hemisphere"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key==ord('4'):
            state["shape"]="Cylinder"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key==ord('5'):
            state["shape"]="Pyramid"
            shape_vertices, edges = SHAPES[state["shape"]]()
            faces = shape_faces(state["shape"])
        elif key in (ord('o'),ord('O')):
            state["projection"]="Orthographic"
        elif key in (ord('p'),ord('P')):
            state["projection"]="Perspective"
        elif key in (ord('s'),ord('S')):
            state["shear"]=not state["shear"]
        elif key in (ord('m'),ord('M')):
            state["order_demo"]=not state["order_demo"]
        elif key in (ord('a'),ord('A')):
            state["auto_z"]=not state["auto_z"]
        elif key in (ord('h'),ord('H')):
            state["hold"]=not state["hold"]
            if state["hold"]:
                state["held_az"]=az
        elif key in (ord('r'),ord('R')):
            state["mouse_x"]=VIEW_W//2
            state["mouse_y"]=H//2
            state["smooth_ax"]=0.0
            state["smooth_ay"]=0.0
            state["held_az"]=0.0
            start=time.perf_counter()

        cv2.imshow("Linear Algebra 3D Visualizer",img)

    # PLUGIN HERE ----------------------------------------------->
    # Stop background video thread cleanly.
    # ------------------------------------------------------------>
    if video_plugin is not None:
        video_plugin.stop()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
