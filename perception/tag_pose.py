"""
perception/tag_pose.py — Görüntü kaynağından BAĞIMSIZ AprilTag algı çekirdeği.
PyBullet'te ArucoCamera, Gazebo'da ROS 2 düğümü aynı iki fonksiyonu çağırır:
  detect_tags(gray, detector)                    -> corners, ids
  solve_pad(corners, ids, K, dist, M_cw, cam_pos) -> pad merkezi (dünya) sözlüğü ya da None
M_cw: kamera (OpenCV: x sağ, y aşağı, z ileri) -> dünya rotasyonu (3x3); cam_pos: kameranın dünya konumu (3,).
"""
import numpy as np
import cv2
from perception.tag_layout import APRILTAG_DICT, TAGS

_REFINE = {"none": cv2.aruco.CORNER_REFINE_NONE, "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
           "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG}


def _objp(size):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def _rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


OBJP   = {tid: _objp(c["size"]) for tid, c in TAGS.items()}
OFFSET = {tid: np.array([c["offset"][0], c["offset"][1], 0.0]) for tid, c in TAGS.items()}


def make_detector(corner_refine="subpix"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, APRILTAG_DICT))
    prm = cv2.aruco.DetectorParameters()
    prm.cornerRefinementMethod = _REFINE[corner_refine]
    return cv2.aruco.ArucoDetector(d, prm)


def detect_tags(gray, detector):
    """Gri görüntüde tag köşelerini bul. Tag yoksa ids None."""
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def solve_pad(corners, ids, K, dist, M_cw, cam_pos):
    """Bulunan tag'lerden pad merkezini çöz (en çok pikselli tag seçilir). Çözüm yoksa None."""
    per_tag = {}
    for k, tid in enumerate(ids.ravel().tolist()):
        if tid not in TAGS:
            continue
        img_pts = corners[k].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(OBJP[tid], img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        t_w = M_cw @ tvec.ravel()                                   # tag merkezi − kamera (dünya)
        X_w = M_cw @ cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0][:, 0]
        yaw = float(np.arctan2(X_w[1], X_w[0]))                     # pad yönü
        rel = t_w - _rotz(yaw) @ OFFSET[tid]                        # pad merkezi − kamera
        if not np.all(np.isfinite(rel)):
            continue
        px = float(np.mean(np.linalg.norm(img_pts - np.roll(img_pts, 1, axis=0), axis=1)))
        per_tag[tid] = dict(rel=rel, px=px, yaw=yaw)
    if not per_tag:
        return None
    best = max(per_tag, key=lambda t: per_tag[t]["px"])
    b = per_tag[best]
    return dict(z=cam_pos + b["rel"], rel=b["rel"], tid=best, px=b["px"], yaw=b["yaw"],
                tags={t: cam_pos + v["rel"] for t, v in per_tag.items()})