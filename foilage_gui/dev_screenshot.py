"""Capture each GUI tab to a PNG without needing window focus (Windows).

    python -m foilage_gui.dev_screenshot [input.json] [outdir]

PrintWindow with PW_RENDERFULLCONTENT renders the window surface even when
occluded, so the developer desktop stays untouched. Used to review the GUI
visually; not part of the app itself.
"""

import ctypes
import ctypes.wintypes as wt
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from foilage_gui.app import FoilageApp  # noqa: E402

UROOT = ctypes.windll.user32
GROOT = ctypes.windll.gdi32


def _capture(hwnd, path):
    rect = wt.RECT()
    UROOT.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc_win = UROOT.GetWindowDC(hwnd)
    hdc_mem = GROOT.CreateCompatibleDC(hdc_win)
    bmp = GROOT.CreateCompatibleBitmap(hdc_win, w, h)
    GROOT.SelectObject(hdc_mem, bmp)
    ok = UROOT.PrintWindow(hwnd, hdc_mem, 2)      # PW_RENDERFULLCONTENT
    buf = np.zeros((h, w, 4), dtype=np.uint8)
    class BMIH(ctypes.Structure):
        _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32)]
    bi = BMIH(ctypes.sizeof(BMIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    GROOT.GetDIBits(hdc_mem, bmp, 0, h, ctypes.c_void_p(buf.ctypes.data),
                    ctypes.byref(bi), 0)         # DIB_RGB_COLORS
    rgb = buf[:, :, [2, 1, 0]]
    import matplotlib.pyplot as plt
    plt.imsave(path, rgb)
    GROOT.DeleteObject(bmp)
    GROOT.DeleteDC(hdc_mem)
    UROOT.ReleaseDC(hwnd, hdc_win)
    return ok


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        REPO / "cases" / "turbine_blade_4" / "input.json"
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / ".gui_shots"
    outdir.mkdir(exist_ok=True)

    root = tk.Tk()
    app = FoilageApp(root, inp.resolve(), threads=6)
    root.update_idletasks()
    root.update()
    hwnd = int(root.frame(), 16) if False else UROOT.GetParent(root.winfo_id())
    UROOT.SetWindowPos(hwnd, 0, 40, 40, 0, 0, 0x0001 | 0x0004)  # move only

    names = ["geometry", "setup", "mesh", "solution", "optimization"]
    for i, name in enumerate(names):
        app.notebook.select(i)
        if name in ("geometry", "mesh", "solution"):
            for _ in range(60):                   # let async builds finish
                root.update()
                time.sleep(0.05)
        else:
            for _ in range(6):
                root.update()
                time.sleep(0.05)
        ok = _capture(hwnd, outdir / f"tab_{name}.png")
        print(f"{name}: {'captured' if ok else 'CAPTURE FAILED'} "
              f"-> {outdir / f'tab_{name}.png'}")
    root.destroy()


if __name__ == "__main__":
    main()
