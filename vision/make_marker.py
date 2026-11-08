"""Platform için ArUco marker görseli üretir (BİR KEZ çalıştırılır).
Tuval 800px = platform üst yüzü 0.60 m'ye eşlenir. Marker 560px (merkez) → fiziksel
kenar = 0.60 * 560/800 = 0.42 m. Bu değer env'deki MARKER_SIZE ile AYNI olmalı.
"""
import cv2, numpy as np, pathlib

DICT      = cv2.aruco.DICT_4X4_50
MARKER_ID = 23
CANVAS    = 800     # px kare tuval  (→ 0.60 m yüz)
MARGIN    = 120     # px beyaz kenar → marker 800-2*120 = 560 px

out = pathlib.Path(__file__).resolve().parent.parent / "assets"
out.mkdir(exist_ok=True)

d = cv2.aruco.getPredefinedDictionary(DICT)
m = cv2.aruco.generateImageMarker(d, MARKER_ID, CANVAS - 2*MARGIN)   # 560x560
canvas = np.full((CANVAS, CANVAS), 255, dtype=np.uint8)              # beyaz zemin
canvas[MARGIN:MARGIN+m.shape[0], MARGIN:MARGIN+m.shape[1]] = m
path = out / "aruco_marker.png"
cv2.imwrite(str(path), canvas)
print("marker ->", path, "| fiziksel marker kenari =",
      round(0.60*(CANVAS-2*MARGIN)/CANVAS, 3), "m")