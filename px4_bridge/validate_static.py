"""
px4_bridge/validate_static.py — Statik algı doğrulaması (Faz 8 Adım C).
Gazebo'daki sabit kameraların karelerinden pad merkezini perception/tag_pose ile çözer, gerçek konumla karşılaştırır.
WSL'de çalışır (ROS 2 + numpy + scipy + OpenCV); torch/PyBullet gerekmez.
"""
import sys, pathlib
PROJ = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(PROJ))
OUT = PROJ / "reports/faz8_c"
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation
from perception.tag_pose import make_detector, detect_tags, solve_pad

# Dünya dosyasındaki kamera pozları (x, y, z, roll, pitch, yaw) — worlds/inis_static.sdf ile AYNI olmalı
CAMS = {"cam_a": (0.00, 0.00, 0.2, 0, 1.5708, 0.0),
        "cam_b": (0.10, -0.05, 0.5, 0, 1.5708, 0.0),
        "cam_c": (0.20, 0.15, 1.0, 0, 1.5708, 0.5)}
PAD_TRUE = np.array([0.0, 0.0, 0.001])
# Gazebo kamera gövdesi (x ileri, y sol, z yukarı) <- OpenCV optik (x sağ, y aşağı, z ileri); sütunlar = optik eksenler
R_LINK_OPT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])


class Validator(Node):
    def __init__(self):
        super().__init__("validate_static")
        self.img, self.info = {}, {}
        for c in CAMS:
            self.create_subscription(Image, f"/{c}/image", lambda m, c=c: self.img.setdefault(c, m), 10)
            self.create_subscription(CameraInfo, f"/{c}/camera_info", lambda m, c=c: self.info.setdefault(c, m), 10)

    def ready(self):
        return all(c in self.img and c in self.info for c in CAMS)


def main():
    rclpy.init(); node = Validator()
    t0 = node.get_clock().now()
    while rclpy.ok() and not node.ready():
        rclpy.spin_once(node, timeout_sec=0.2)
        if (node.get_clock().now() - t0).nanoseconds > 20e9:
            print("20 s icinde kare gelmedi. Eksik:", [c for c in CAMS if c not in node.img or c not in node.info]); return
    OUT.mkdir(parents=True, exist_ok=True)
    det = make_detector("subpix")
    worst = 0.0
    for c, (x, y, z, r, p, yw) in CAMS.items():
        m, ci = node.img[c], node.info[c]
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(str(OUT / f"{c}.png"), gray)
        K = np.array(ci.k, dtype=np.float64).reshape(3, 3)              # iç parametreler Gazebo'dan (elle değil)
        dist = np.array(ci.d, dtype=np.float64) if len(ci.d) else np.zeros(5)
        M_cw = Rotation.from_euler("xyz", [r, p, yw]).as_matrix() @ R_LINK_OPT   # optik -> dünya
        corners, ids = detect_tags(gray, det)
        res = None if ids is None else solve_pad(corners, ids, K, dist, M_cw, np.array([x, y, z]))
        if res is None:
            print(f"{c}: h={z:.1f} m  TAG BULUNAMADI (ids={None if ids is None else ids.ravel().tolist()})"); worst = np.inf; continue
        e = res["z"] - PAD_TRUE
        e_xy, e_z = np.linalg.norm(e[:2]), abs(e[2])
        worst = max(worst, e_xy)
        print(f"{c}: h={z:.1f} m  K: fx={K[0,0]:.1f} cx={K[0,2]:.1f} | tag'ler={sorted(res['tags'])} secilen={res['tid']} "
              f"| pad tahmini={np.round(res['z'], 3)} hata yatay={100*e_xy:.1f} cm dikey={100*e_z:.1f} cm "
              f"| pad yaw={np.degrees(res['yaw']):.1f} derece")
    print("EN BUYUK YATAY HATA:", f"{100*worst:.1f} cm", "✔" if worst < 0.03 else "✘", f"| kareler: {OUT}")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()