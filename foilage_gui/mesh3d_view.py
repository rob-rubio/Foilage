"""GPU 3D view of the outer surface of a 3D wedge mesh (moderngl).

Renders the boundary (marker) faces of an SU2 3D mesh - hub/shroud
streamtube walls, periodic side faces, blade, inlet/outlet - with
per-marker toggles, a wireframe overlay and translucent hub/shroud so the
blade is visible inside the wedge. Rendering happens offscreen with
moderngl (standalone context) and the frame is blitted into the Tk window
as an image, so no OpenGL/Tkinter interop is required.

Drag = orbit, wheel = zoom, R or the button = reset view. Frames render
at 1x while dragging and re-render at 2x supersampling when idle.

Usage from the Mesh tab:
    Mesh3DView.open(app, path)     # guards deps + double-open
"""

import math
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import moderngl
except ImportError:                       # GUI must keep working without it
    moderngl = None
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

# display colors (linear-ish RGB) per marker; hub/shroud form the
# translucent outer shell of the wedge
MARKER_COLORS = {
    "airfoil": (0.87, 0.87, 0.90),
    "inlet": (0.30, 0.55, 0.92),
    "outlet": (0.90, 0.32, 0.32),
    "periodic_bottom": (0.30, 0.78, 0.42),
    "periodic_top": (0.58, 0.88, 0.48),
    "hub": (0.74, 0.56, 0.34),
    "shroud": (0.88, 0.62, 0.80),
    "farfield": (0.60, 0.45, 0.82),
}
FALLBACK_COLOR = (0.70, 0.70, 0.70)
SHELL_MARKERS = ("hub", "shroud")

_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_inflate;
out vec3 v_nrm;
out vec3 v_col;
void main() {
    vec3 p = u_center + (in_pos - u_center) * u_inflate;
    v_nrm = in_nrm;
    v_col = in_col;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

_FRAG = """
#version 330
in vec3 v_nrm;
in vec3 v_col;
uniform vec3 u_light;
uniform float u_alpha;
out vec4 frag;
void main() {
    float diff = abs(dot(normalize(v_nrm), u_light));
    float shade = 0.42 + 0.58 * diff;
    frag = vec4(v_col * shade, u_alpha);
}
"""


def is_3d_file(path):
    """True when the SU2 mesh file declares NDIME= 3."""
    try:
        with open(path) as f:
            first = f.readline().strip()
    except OSError:
        return False
    return first.startswith("NDIME=") and first.split("=")[1].split()[0] == "3"


def parse_boundary(path):
    """Outer surface of a 3D SU2 mesh: points + marker faces.

    The volume-element block is skipped without tokenizing (the solver
    meshes hold >500k volume elements), so even ~70 MB files parse in a
    couple of seconds. Returns {"points": (n,3) float32, "markers":
    OrderedDict[name -> list of faces], "volume_elems": int}. Raises
    ValueError for 2D files.
    """
    t0 = time.time()
    with open(path) as f:
        lines = f.read().splitlines()
    points = None
    markers = OrderedDict()
    volume_elems = 0
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith("NDIME="):
            if s.split("=")[1].split()[0] != "3":
                raise ValueError(
                    f"{Path(path).name}: not a 3D mesh (NDIME= "
                    f"{s.split('=')[1].split()[0]})")
            i += 1
        elif s.startswith("NPOIN="):
            cnt = int(s.split("=")[1].split()[0])
            pts = np.empty((cnt, 3), dtype=np.float32)
            for j in range(cnt):
                t = lines[i + 1 + j].split()
                pts[j, 0], pts[j, 1], pts[j, 2] = \
                    float(t[0]), float(t[1]), float(t[2])
            points = pts
            i += cnt + 1
        elif s.startswith("NELEM="):
            volume_elems = int(s.split("=")[1].split()[0])
            i += volume_elems + 1            # skip the whole block
        elif s.startswith("MARKER_TAG="):
            name = s.split("=", 1)[1].strip()
            cnt = int(lines[i + 1].split("=")[1].split()[0])
            faces = []
            for j in range(cnt):
                t = [int(x) for x in lines[i + 2 + j].split()]
                etype, nodes = t[0], t[1:]
                if etype == 9:
                    faces.append(nodes[:4])
                elif etype == 5:
                    faces.append(nodes[:3])
                else:
                    faces.append(nodes)      # tolerate exotic face types
            if faces:
                markers[name] = faces
            i += cnt + 2
        else:
            i += 1
    if points is None:
        raise ValueError(f"{Path(path).name}: no NPOIN block found")
    return {"points": points, "markers": markers,
            "volume_elems": volume_elems, "parse_s": time.time() - t0}


# ------------------------------------------------------------- camera math
def _perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, np.asarray(up, dtype=np.float64))
    sn = np.linalg.norm(s)
    if sn < 1e-9:                       # view dir parallel to up: pick a
        s = np.cross(f, (1.0, 0.0, 0.0))  # safe secondary axis
        sn = np.linalg.norm(s)
        if sn < 1e-9:
            s = np.cross(f, (0.0, 1.0, 0.0))
            sn = np.linalg.norm(s)
    s /= sn
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


class OuterSurfaceRenderer:
    """Offscreen moderngl renderer for one parsed boundary mesh."""

    def __init__(self, parsed):
        if moderngl is None or Image is None:
            raise RuntimeError("moderngl / Pillow not available")
        self.ctx = moderngl.create_context(standalone=True)
        self.prog = self.ctx.program(vertex_shader=_VERT,
                                     fragment_shader=_FRAG)
        self.center = parsed["points"].mean(axis=0).astype(np.float64)
        self._span = (parsed["points"].max(axis=0)
                      - parsed["points"].min(axis=0))
        self.radius = float(np.linalg.norm(self._span)) / 2.0 or 1.0
        self.tris = {}
        self.lines = {}
        self.meta = {}
        for name, faces in parsed["markers"].items():
            col = np.array(MARKER_COLORS.get(name, FALLBACK_COLOR),
                           dtype=np.float32)
            self.tris[name] = self._tri_buffer(faces,
                                               parsed["points"], col)
            self.lines[name] = self._line_buffer(faces,
                                                 parsed["points"], col * 0.45)
            _vao, vbo, _n = self.tris[name]
            rec = np.frombuffer(vbo.read(), dtype=np.float32).reshape(-1, 9)
            self.meta[name] = {"mean": rec[:, :3].mean(axis=0),
                               "nrm": rec[:, 3:6].mean(axis=0)}
        self._fbo = None
        self._size = (0, 0)

    def _tri_buffer(self, faces, points, col):
        verts = np.empty((len(faces) * 6, 9), dtype=np.float32)  # quad=2 tris
        k = 0
        for f in faces:
            p = points[[f[0], f[1], f[2]]]
            nrm = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(nrm)
            nrm = nrm / nn if nn > 1e-20 else np.zeros(3, np.float32)
            tris = (p, points[[f[0], f[2], f[3]]]) if len(f) == 4 else (p,)
            for tri in tris:
                for v in tri:
                    verts[k, :3] = v
                    verts[k, 3:6] = nrm
                    verts[k, 6:] = col
                    k += 1
        vbo = self.ctx.buffer(verts[:k].tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f",
                                                 "in_pos", "in_nrm",
                                                 "in_col")])
        return vao, vbo, k // 3

    def _line_buffer(self, faces, points, col):
        segs = []
        for f in faces:
            for a in range(len(f)):
                b = (a + 1) % len(f)
                segs.append((f[a], f[b]))
        verts = np.empty((len(segs) * 2, 9), dtype=np.float32)
        k = 0
        for a, b in segs:
            for nid in (a, b):
                verts[k, :3] = points[nid]
                verts[k, 3:6] = 1.0     # any unit normal: lines shade flat
                verts[k, 6:] = col
                k += 1
        vbo = self.ctx.buffer(verts.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f",
                                                 "in_pos", "in_nrm",
                                                 "in_col")])
        return vao, vbo, len(segs)

    def _shells_by_depth(self, eye):
        """hub/shroud ordered far-to-near from the camera (blend order).

        The wall whose mean face normal faces the camera is the near one;
        both statistics are cached at buffer-build time."""
        order = []
        for name in SHELL_MARKERS:
            meta = self.meta.get(name)
            if meta is None:
                continue
            facing = float(np.dot(meta["nrm"], eye - meta["mean"]))
            order.append((facing, name))     # larger = nearer
        order.sort()                         # far (negative) first
        return [name for _f, name in order]

    def fit_distance(self, az, el, aspect, fill=0.78):
        """Camera distance framing the whole object at ~`fill` of the view.

        Projects the bounding-box corners with a pinhole model and rescales
        the distance until the largest |NDC| extent is ~fill (two
        iterations converge amply). Fitting the projected extent instead
        of the box diagonal keeps thin wedge slivers from looking tiny."""
        corners = np.array([[x, y, z] for x in (0.0, 1.0)
                            for y in (0.0, 1.0) for z in (0.0, 1.0)])
        corners = self.center + (corners - 0.5) * self._span
        azr, elr = math.radians(az), math.radians(el)
        dirv = np.array([math.cos(elr) * math.cos(azr),
                         math.cos(elr) * math.sin(azr),
                         math.sin(elr)])
        f = 1.0 / math.tan(math.radians(38.0) / 2.0)
        dist = max(self.radius, 1e-6)
        for _ in range(3):
            rel = corners - (self.center + dist * dirv)
            view = rel @ np.column_stack(
                self._view_axes(dirv))      # world -> view coords
            ndc_x = f * view[:, 0] / -view[:, 2] / aspect
            ndc_y = f * view[:, 1] / -view[:, 2]
            m = max(np.abs(ndc_x).max(), np.abs(ndc_y).max())
            if m < 1e-6:
                break
            dist = dist * m / fill
        return max(dist, self.radius * 0.1)

    @staticmethod
    def _view_axes(dirv):
        """Right/up basis for a view along -dirv with world up +Z."""
        f = -dirv                          # forward (towards target)
        s = np.cross(f, (0.0, 0.0, 1.0))
        if np.linalg.norm(s) < 1e-9:
            s = np.cross(f, (1.0, 0.0, 0.0))
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        return s, u, -f                    # right, up, backward(+z_view)

    def render(self, w, h, az, el, dist, visible, shell_alpha=0.35,
               wireframe=True, ssaa=2):
        """Render one frame; returns a PIL RGB image of size (w, h)."""
        w = max(int(w), 2)
        h = max(int(h), 2)
        fw, fh = w * ssaa, h * ssaa
        if self._fbo is None or self._size != (fw, fh):
            if self._fbo is not None:
                self._fbo.depth_attachment.release()
                self._fbo.color_attachments[0].release()
                self._fbo.release()
            tex = self.ctx.texture((fw, fh), 4)
            depth = self.ctx.depth_renderbuffer((fw, fh))
            self._fbo = self.ctx.framebuffer(tex, depth)
            self._size = (fw, fh)

        azr, elr = math.radians(az), math.radians(el)
        eye = self.center + dist * np.array(
            [math.cos(elr) * math.cos(azr),
             math.cos(elr) * math.sin(azr),
             math.sin(elr)])
        proj = _perspective(38.0, w / h, dist * 0.02, dist * 20.0)
        view = _look_at(eye, self.center, (0.0, 0.0, 1.0))
        # OpenGL expects column-major: upload the transpose of the math
        # matrix so the shader receives (proj @ view) untransposed
        mvp = np.ascontiguousarray((proj @ view).T.astype(np.float32))
        light = (self.center - eye)
        light /= np.linalg.norm(light)

        u = self.prog
        u["u_mvp"].write(mvp.tobytes())
        u["u_center"].write(self.center.astype(np.float32).tobytes())
        u["u_light"].write(light.astype(np.float32).tobytes())
        u["u_inflate"].value = 1.0
        u["u_alpha"].value = 1.0

        self._fbo.use()
        self.ctx.clear(0.09, 0.10, 0.13)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        for name in self.tris:
            if name not in visible or name in SHELL_MARKERS:
                continue
            self.tris[name][0].render(moderngl.TRIANGLES)
        if shell_alpha >= 0.999:
            for name in self._shells_by_depth(eye):
                if name in self.tris and name in visible:
                    self.tris[name][0].render(moderngl.TRIANGLES)
        else:
            # translucent shells: moderngl has no depth-mask control, so
            # the two walls are drawn far-to-near for correct blending
            self.ctx.enable(moderngl.BLEND)
            u["u_alpha"].value = float(shell_alpha)
            for name in self._shells_by_depth(eye):
                if name in self.tris and name in visible:
                    self.tris[name][0].render(moderngl.TRIANGLES)
            u["u_alpha"].value = 1.0
        if wireframe:
            u["u_inflate"].value = 1.002      # lift lines off the surface
            u["u_alpha"].value = 0.55
            for name in self.lines:
                if name in visible:
                    self.lines[name][0].render(moderngl.LINES)
            u["u_inflate"].value = 1.0
            u["u_alpha"].value = 1.0
        self.ctx.disable(moderngl.BLEND)

        data = self._fbo.read(components=3, alignment=1)
        img = Image.frombuffer("RGB", (fw, fh), data, "raw", "RGB", 0, 1)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if ssaa != 1:
            img = img.resize((w, h), Image.BILINEAR)
        return img

    def release(self):
        for bufs in list(self.tris.values()) + list(self.lines.values()):
            bufs[0].release()
            bufs[1].release()
        self.tris.clear()
        self.lines.clear()
        if self._fbo is not None:
            self._fbo.depth_attachment.release()
            self._fbo.color_attachments[0].release()
            self._fbo.release()
            self._fbo = None
        self.ctx.release()


class Mesh3DView(tk.Toplevel):
    """Window showing the mesh's outer surface; orbit/zoom with the mouse."""

    IDLE_MS = 180          # after last drag -> crisp supersampled frame

    def __init__(self, app, path):
        super().__init__(app.root)
        self.app = app
        self.path = Path(path)
        self.title(f"3D outer view - {self.path.name}")
        self.geometry("1020x720")
        self.renderer = None
        self._q = queue.Queue()
        self._photo = None
        self._idle_job = None
        self._resize_job = None
        self.az, self.el = -60.0, 18.0
        self.dist = None
        self._drag = None

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        self.marker_vars = {}
        self.wire_var = tk.BooleanVar(value=True)
        self.shell_var = tk.IntVar(value=35)
        ttk.Button(bar, text="Reset view", command=self._reset_view).pack(
            side="left", padx=2)
        ttk.Checkbutton(bar, text="wireframe",
                        variable=self.wire_var,
                        command=lambda: self._request(hq=True)).pack(
            side="left", padx=8)
        ttk.Label(bar, text="shell opacity %").pack(side="left", padx=(8, 2))
        sc = ttk.Scale(bar, from_=0, to=100, variable=self.shell_var,
                       command=lambda _v: self._request(hq=False))
        sc.pack(side="left")
        sc.configure(length=110)
        self._marker_row = ttk.Frame(bar)
        self._marker_row.pack(side="left", padx=10)
        self.info_var = tk.StringVar(value=f"parsing {self.path.name} ...")
        ttk.Label(self, textvariable=self.info_var, anchor="w",
                  font=("Consolas", 8)).pack(fill="x", padx=6, pady=(0, 2))

        self.img_label = tk.Label(self, bg="#171a20")
        self.img_label.pack(fill="both", expand=True)

        self.img_label.bind("<Button-1>", self._press)
        self.img_label.bind("<B1-Motion>", self._motion)
        self.img_label.bind("<ButtonRelease-1>", self._release)
        self.img_label.bind("<MouseWheel>", self._wheel)
        self.bind("<r>", self._reset_view)
        self.bind("<Configure>", self._configure)
        self.protocol("WM_DELETE_WINDOW", self._close)

        threading.Thread(target=self._worker, daemon=True,
                         name="mesh3d-parse").start()
        self.after(60, self._poll)

    # ------------------------------------------------------------ loading
    def _worker(self):
        try:
            parsed = parse_boundary(self.path)
            self._q.put(("ok", parsed))
        except Exception as e:
            self._q.put(("error", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            status, payload = self._q.get_nowait()
        except queue.Empty:
            self.after(60, self._poll)
            return
        if status == "error":
            self.info_var.set(f"failed to read {self.path.name}: {payload}")
            return
        try:
            self.renderer = OuterSurfaceRenderer(payload)
        except Exception as e:
            self.info_var.set(f"OpenGL unavailable: {e} "
                              "(moderngl needs a working GPU driver)")
            return
        self._build_marker_toggles(payload["markers"])
        n_pts = len(payload["points"])
        n_faces = sum(len(f) for f in payload["markers"].values())
        self.info_var.set(
            f"{self.path.name}: {n_pts} nodes, {n_faces} boundary faces, "
            f"{payload['volume_elems']} volume cells "
            f"(parsed in {payload['parse_s']:.1f}s) | drag = orbit, "
            f"wheel = zoom, R = reset")
        self._reset_view()

    def _build_marker_toggles(self, markers):
        for name in markers:
            var = tk.BooleanVar(value=True)
            col = MARKER_COLORS.get(name, FALLBACK_COLOR)
            hexcol = "#%02x%02x%02x" % tuple(
                int(max(0.0, min(1.0, c)) * 255) for c in col)
            cb = tk.Checkbutton(self._marker_row, text=name, variable=var,
                                fg=hexcol, font=("TkDefaultFont", 8),
                                command=lambda: self._request(hq=True))
            cb.pack(side="left", padx=(6, 0))
            self.marker_vars[name] = var

    # ------------------------------------------------------------ camera
    def _reset_view(self, *_):
        self.az, self.el = -75.0, 25.0
        if self.renderer is not None:
            w = max(self.img_label.winfo_width(), 200)
            h = max(self.img_label.winfo_height(), 150)
            self.dist = self.renderer.fit_distance(
                self.az, self.el, w / h, fill=0.78)
        self._request(hq=True)

    def _visible(self):
        return {n for n, v in self.marker_vars.items() if v.get()}

    # ------------------------------------------------------------ events
    def _press(self, e):
        self._drag = (e.x, e.y, self.az, self.el)

    def _motion(self, e):
        if self._drag is None:
            return
        x0, y0, az0, el0 = self._drag
        self.az = az0 - (e.x - x0) * 0.4
        self.el = max(-89.0, min(89.0, el0 + (e.y - y0) * 0.4))
        self._request(hq=False)

    def _release(self, _e):
        self._drag = None
        self._request(hq=True)

    def _wheel(self, e):
        if self.dist is None:
            return
        factor = 1.1 if e.delta > 0 else 1 / 1.1
        self.dist = min(max(self.dist * factor,
                            self.renderer.radius * 0.05),
                        self.renderer.radius * 40.0)
        self._request(hq=False)

    def _configure(self, e):
        if e.widget is not self:
            return
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, lambda: self._request(hq=True))

    # ------------------------------------------------------------ render
    def _request(self, hq):
        if self.renderer is None or self.dist is None:
            return
        self._draw(ssaa=2 if hq else 1)
        if self._idle_job is not None:
            self.after_cancel(self._idle_job)
        if not hq:
            self._idle_job = self.after(
                self.IDLE_MS, lambda: self._draw(ssaa=2))

    def _draw(self, ssaa):
        if self.renderer is None or self.dist is None:
            return
        w = max(self.img_label.winfo_width(), 64)
        h = max(self.img_label.winfo_height(), 64)
        try:
            img = self.renderer.render(
                w, h, self.az, self.el, self.dist,
                self._visible() or set(self.marker_vars),
                shell_alpha=self.shell_var.get() / 100.0,
                wireframe=self.wire_var.get(), ssaa=ssaa)
        except Exception as e:
            self.info_var.set(f"render error: {e}")
            return
        self._photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._photo)

    def _close(self):
        if self.renderer is not None:
            try:
                self.renderer.release()
            except Exception:
                pass
        self.destroy()

    # ------------------------------------------------------------ opener
    _open = None

    @classmethod
    def open(cls, app, path):
        """Open (or raise) the 3D view for `path`; friendly errors only."""
        if moderngl is None or Image is None or ImageTk is None:
            messagebox.showerror(
                "3D view unavailable",
                "The 3D mesh view needs moderngl and Pillow in the GUI "
                "interpreter. Install them with:\n\n"
                "uv pip install --python <gui-python> moderngl pillow",
                parent=app.root)
            return
        win = cls._open
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return
        win = cls(app, path)
        cls._open = win
        win.protocol("WM_DELETE_WINDOW", win._close_and_forget)
        return win

    def _close_and_forget(self):
        Mesh3DView._open = None
        self._close()
