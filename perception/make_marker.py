"""
make_marker.py — Nested AprilTag iniş pedi görseli üretir (BİR KEZ çalıştırılır).
assets/aruco_marker.png: 800px tuval (= 0.60 m pad), iki AprilTag (büyük+küçük)
tag_layout.py'deki ofset/boyutlarda, beyaz quiet-zone'larla.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from perception.tag_layout import CANVAS_PX, PAD_M, PXPM, APRILTAG_DICT, TAGS

d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, APRILTAG_DICT.replace("DICT_", "DICT_")))
canvas = np.full((CANVAS_PX, CANVAS_PX), 255, dtype=np.uint8)   # beyaz pad

def world_to_px(x, y):
    """pad (x,y) m -> (col,row) px. x sağ, y yukarı (row yukarı = küçük)."""
    col = (x + PAD_M/2) * PXPM
    row = (PAD_M/2 - y) * PXPM
    return int(round(col)), int(round(row))

for tid, cfg in TAGS.items():
    s_px = int(round(cfg["size"] * PXPM))
    img  = cv2.aruco.generateImageMarker(d, tid, s_px)
    cx, cy = world_to_px(*cfg["offset"])
    r0, c0 = cy - s_px//2, cx - s_px//2
    canvas[r0:r0+s_px, c0:c0+s_px] = img
    print(f"tag {tid}: boyut={cfg['size']} m ({s_px}px), ofset={cfg['offset']} -> px merkez=({cx},{cy})")

out = pathlib.Path(__file__).resolve().parent.parent / "assets" / "aruco_marker.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), canvas)
print("nested marker ->", out)
