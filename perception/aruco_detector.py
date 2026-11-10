"""
perception/aruco_detector.py — Nested AprilTag tabanlı platform algısı (gerçekçi kamera modeli).

Drona monte, GÖVDEYE SABİT, aşağı bakan kamera ile pedin AprilTag'lerini (büyük+küçük,
tag_layout.py) tespit eder ve PAD MERKEZİNİN konumunu çözer (solvePnP).

Ölçüm zinciri (gerçek bir companion-bilgisayar boru hattını taklit eder):
  1. Render: kamera dronun GERÇEK pozundan görür (fizik).
  2. Görüntü bozulması (cfg["degrade"]): optik bulanıklık + hareket bulanıklığı (pozlama
     süresince piksel kayması), bölüm-sabit pozlama kazancı + kare-kare titreşim, sensör gürültüsü.
  3. Tespit + solvePnP → tag'in KAMERA çerçevesindeki konumu ve yönü.
  4. GÖRELİ ölçüm: pad merkezi − kamera, dünya eksenlerinde. Kamera→dünya dönüşümü için
     dronun TAHMİNİ tutumu kullanılır (IMU: bölüm-sabit sapma + kare gürültüsü). Tag ofseti
     pedin yönüne (yaw, tag'den okunur) göre döndürülerek çıkarılır → dönen platformda da doğru.
     PX4 LANDING_TARGET de bu göreli bilgiyi bekler.
  5. Mutlak tahmin z = kamera_konumu + göreli. Dronun kendi konumu bilinir kabul edilir
     (kontrolcü de bunu kullanıyor; gerçekte GNSS/VIO).

Zamanlama (kamera hızı, gecikme) bu sınıfta DEĞİL, env'deki ölçüm kuyruğundadır.

Öz-test:  python perception/aruco_detector.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # 'perception' paketi için
import numpy as np
import pybullet as p
import cv2
from perception.tag_layout import APRILTAG_DICT, TAGS
from perception.tag_pose import make_detector, detect_tags, solve_pad


# ----------------------------- kamera konfigürasyonları -----------------------------
# ideal     : eski davranış (her kontrol adımında, gecikmesiz, kusursuz görüntü).
# realistic : küçük companion-bilgisayarlı dron (RPi/Jetson sınıfı) için tipik değerler.
CAM_PRESETS = {
    "ideal": dict(
        hz=None,                 # None = her kontrol adımında kare
        latency_s=0.0,           # yakalama → kontrolcüye ulaşma
        latency_jitter_s=0.0,    # gecikmenin ± düzgün dağılımlı oynaması
        degrade=False,           # görüntü bozulmaları
        exposure_s=0.0,          # pozlama süresi (hareket bulanıklığı)
        optics_sigma=0.0,        # lens/odak bulanıklığı (px, Gauss sigma)
        noise_std=0.0,           # sensör gürültüsü (gri seviye)
        gain_range=(1.0, 1.0),   # bölüm-sabit pozlama kazancı (ışık koşulu)
        gain_jitter=0.0,         # kare-kare kazanç oynaması (oransal std)
        att_bias_deg=0.0,        # IMU tutum sapması (bölüm-sabit, eksen başı std)
        att_noise_deg=0.0,       # IMU tutum gürültüsü (kare başı, eksen başı std)
        corner_refine="subpix",  # köşe iyileştirme: none | subpix | apriltag (subpix: alçakta 5cm→0.1cm)
    ),
    "realistic": dict(
        hz=15.0,
        latency_s=0.10,
        latency_jitter_s=0.02,
        degrade=True,
        exposure_s=0.008,
        optics_sigma=0.6,
        noise_std=4.0,
        gain_range=(0.6, 1.3),
        gain_jitter=0.05,
        att_bias_deg=0.5,
        att_noise_deg=0.2,
        corner_refine="subpix",
    ),
}


def cam_config(cfg=None):
    """None → ideal; 'realistic'/'ideal' → hazır ayar; dict → (varsa 'base' preset) üstüne yazar."""
    if cfg is None:
        return dict(CAM_PRESETS["ideal"])
    if isinstance(cfg, str):
        return dict(CAM_PRESETS[cfg])
    cfg = dict(cfg)
    out = dict(CAM_PRESETS[cfg.pop("base", "ideal")])
    unknown = set(cfg) - set(out)
    if unknown:
        raise KeyError(f"bilinmeyen kamera ayari: {unknown}")
    out.update(cfg)
    return out



def _small_rot(rvec):
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]



class ArucoCamera:
    def __init__(self, client, drone_id, marker_id=None, marker_size=None,
                 res=480, vfov_deg=120.0, mount_offset=(0.0, 0.0, -0.02), gimbal=False,
                 cfg=None, rng=None):
        # marker_id/marker_size geriye-uyum için kabul edilir ama artık tag_layout kullanılır.
        # gimbal=True: kamera dron eğiminden bağımsız HEP dik aşağı bakar (nadir-stabilize).
        self.client, self.drone_id = client, drone_id
        self.W = self.H = int(res)
        self.vfov = float(vfov_deg)
        self.mount = np.array(mount_offset, dtype=float)
        self.gimbal = bool(gimbal)
        self.cfg = cam_config(cfg)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.f = (self.H / 2.0) / np.tan(np.radians(self.vfov) / 2.0)
        self.K = np.array([[self.f, 0, self.W/2.0], [0, self.f, self.H/2.0], [0, 0, 1.0]], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)

        self.detector = make_detector(self.cfg["corner_refine"])

        self.proj = p.computeProjectionMatrixFOV(self.vfov, 1.0, 0.05, 10.0, physicsClientId=client)

        # bölüm-sabit büyüklükler (kamera her reset'te yeniden kurulur)
        c = self.cfg
        self.gain = float(self.rng.uniform(*c["gain_range"]))
        self.R_att_bias = _small_rot(self.rng.normal(0.0, np.radians(c["att_bias_deg"]), 3))

        self.last_frame = None     # teşhis için son (bozulmuş) gri kare

    # ------------------------------------------------------------------ geometri
    def _drone_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return np.array(pos), R

    def _axes(self, pos, R):
        """Kamera merkezi + bakış (f) ve yukarı (u) yönleri, verilen gövde tutumu R için."""
        eye = pos + R @ self.mount
        if self.gimbal:
            f = np.array([0.0, 0.0, -1.0]); u = np.array([0.0, 1.0, 0.0])
        else:
            f = R @ np.array([0.0, 0.0, -1.0]); u = R @ np.array([0.0, 1.0, 0.0])
        return eye, f/np.linalg.norm(f), u/np.linalg.norm(u)

    @staticmethod
    def _cam_to_world(f, u):
        """Kamera (OpenCV: x sağ, y aşağı, z ileri) → dünya rotasyonu (sütunlar = kamera eksenleri)."""
        x = np.cross(f, u); x /= np.linalg.norm(x)
        return np.column_stack([x, -u, f])

    # ------------------------------------------------------------------ görüntü
    def _render_gray(self, eye, f, u):
        view = p.computeViewMatrix(eye.tolist(), (eye+f).tolist(), u.tolist(), physicsClientId=self.client)
        _, _, rgb, _, _ = p.getCameraImage(self.W, self.H, view, self.proj,
                                           renderer=p.ER_TINY_RENDERER,   # başsız modda güvenilir
                                           flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.client)
        rgb = np.reshape(np.array(rgb, dtype=np.uint8), (self.H, self.W, 4))[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _degrade(self, gray, eye):
        """Optik + hareket bulanıklığı, pozlama kazancı, sensör gürültüsü."""
        c = self.cfg
        lin, ang = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        h = max(float(eye[2]), 0.15)                       # yere göre yükseklik (kaba)
        # pozlama süresince görüntüdeki kayma (px): dönme + öteleme kaynaklı akış
        smear = self.f * c["exposure_s"] * (np.linalg.norm(ang[:2]) + np.linalg.norm(lin[:2]) / h)
        sigma = np.sqrt(c["optics_sigma"]**2 + (smear / 2.0)**2)
        img = gray.astype(np.float32)
        if sigma > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        g = self.gain * (1.0 + c["gain_jitter"] * self.rng.standard_normal())
        img = img * g
        if c["noise_std"] > 0:
            img = img + self.rng.normal(0.0, c["noise_std"], img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8), float(smear)

    # ------------------------------------------------------------------ ölçüm
    def measure(self):
        """Bir kare yakala ve ölç. Tag yoksa None; varsa sözlük:
           z     : pad merkezinin mutlak (dünya) tahmini (3,)
           rel   : pad merkezi − kamera, dünya eksenlerinde (3,)  [PX4 LANDING_TARGET gibi]
           tid   : kullanılan tag, px: tag'in görüntüdeki kenar uzunluğu (px), yaw: pad yönü (rad)
           tags  : {tid: z} (teşhis), smear: hareket bulanıklığı (px)"""
        c = self.cfg
        pos, R_true = self._drone_pose()
        eye, f, u = self._axes(pos, R_true)
        gray = self._render_gray(eye, f, u)
        smear = 0.0
        if c["degrade"]:
            gray, smear = self._degrade(gray, eye)
        self.last_frame = gray
        corners, ids = detect_tags(gray, self.detector)
        if ids is None:
            return None
        # Kamera==> dünya dönüşümü TAHMİNİ tutumla (IMU sapma + gürültü) — rastgele sayı sırası eskisiyle aynı
        R_est = R_true @ self.R_att_bias @ _small_rot(self.rng.normal(0.0, np.radians(c["att_noise_deg"]), 3))
        eye_est, f_e, u_e = self._axes(pos, R_est)
        M_cw = self._cam_to_world(f_e, u_e)
        out = solve_pad(corners, ids, self.K, self.dist, M_cw, eye_est)
        if out is None:
            return None
        out["smear"] = smear
        return out


# --------------------- öz-test ---------------------
if __name__ == "__main__":
    from env.landing_aviary import LandingAviary
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    av = LandingAviary(gui=False, act=ActionType.PID, obs=ObservationType.KIN)
    av.reset(); av._move_platform()
    true_pos, _ = av._platform_state(av.step_counter / av.PYB_FREQ)
    print("gercek platform:", np.round(true_pos, 3))

    def run(cfg, pad_yaw_deg, tilt_deg=0.0):
        cam = ArucoCamera(av.CLIENT, av.DRONE_IDS[0], cfg=cfg, rng=np.random.default_rng(0))
        q_pad = p.getQuaternionFromEuler([0, 0, np.radians(pad_yaw_deg)])
        p.resetBasePositionAndOrientation(av.MARKER_BODY, [true_pos[0], true_pos[1], true_pos[2] + 0.001],
                                          q_pad, physicsClientId=av.CLIENT)
        q_dr = p.getQuaternionFromEuler([np.radians(tilt_deg), 0, 0])
        print(f"\n[{cfg}] pad yaw={pad_yaw_deg} deg, dron roll={tilt_deg} deg")
        print(f"{'yukseklik':>9} {'hata(cm)':>9} {'tag':>4} {'px':>6} {'yaw(deg)':>9}")
        for z in [1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15]:
            p.resetBasePositionAndOrientation(av.DRONE_IDS[0],
                                              [true_pos[0], true_pos[1], true_pos[2] + z],
                                              q_dr, physicsClientId=av.CLIENT)
            m = cam.measure()
            if m is None:
                print(f"{z:>9.2f} {'-':>9}")
            else:
                err = np.linalg.norm(m["z"] - true_pos) * 100
                print(f"{z:>9.2f} {err:>9.2f} {m['tid']:>4} {m['px']:>6.1f} {np.degrees(m['yaw']):>9.1f}")

    run("ideal", 0)
    run("ideal", 30)
    run("realistic", 0)
    run("realistic", 30, tilt_deg=8)
    av.close()
