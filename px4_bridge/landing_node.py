"""
px4_bridge/landing_node.py — Faz 8 Adım D: PX4 SITL + Gazebo'da otonom iniş düğümü (ROS 2).

Zincir (LandingAviary ile AYNI sıra, 30 Hz):
  kamera karesi (15 Hz) -> tag_pose -> IMM -> [gözlem] -> politika (numpy) -> kapı -> güdüm -> hedef + 0.05·aksiyon
  -> PX4 offboard konum setpoint'i.
Her şey "PyBullet çerçevesinde" hesaplanır: ENU, orijin = pad'in başlangıç konumu − (0, 0, PLATFORM_TOP),
dronun referans noktası = AYAK hizası (x500 gövde merkezi yerden ~0.23 m yukarıda; Crazyflie'de birkaç cm).

Deney durum makinesi (iniş mantığından ayrı):
  BEKLE -> (offboard + arm) -> KALKIS -> BASLANGIC (pad üstü başlangıç noktasında 3 s bekle, orijini kur) -> INIS -> BITTI
Kullanım (🐧, `ros` sonrası, px4_bridge/ içinden):  python3 landing_node.py [--no-rl] [--tag ad] [--dx 0.3 --dy -0.2 --h 0.8]
"""
import sys, pathlib, argparse, csv, time, math
PROJ = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(PROJ))
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry,
                          VehicleStatus, VehicleLandDetected)
from scipy.spatial.transform import Rotation

from perception.tag_pose import make_detector, detect_tags, solve_pad
from estimation.imm_ekf import PlatformIMM
from guidance.descent_gate import DescentGate
from guidance.landing_guidance import LandingGuidance
from px4_bridge.frames import enu_from_ned, ned_from_enu, R_from_q_wxyz, R_enu_flu_from_R_ned_frd
from px4_bridge.obs_builder import ObsBuilder
from px4_bridge.policy_np import PolicyNP

# --- env ile aynı sabitler (env/landing_aviary.py) ---
CTRL_HZ      = 30
DT           = 1.0 / CTRL_HZ
CAM_HZ       = 15            # realistic kamera ön ayarı
RES_SCALE    = 0.05
T_LAND       = 8.0
EP_LEN       = 12.0
PLATFORM_TOP = 0.07          # PyBullet'te pad üst yüzeyi (m)
MEAS_STD     = 0.03
WAVE_HZ      = 0.3           # IMM salınım modu: sabit pad'de önemsiz; gerçek frekans bilinmiyor (N12)

# --- x500_inis geometrisi (px4_bridge/models/x500_inis/model.sdf) ---
FOOT  = 0.22                                         # gövde merkezi -> ayak hizası (m)
MOUNT = np.array([0.0, 0.0, -0.05])                  # kamera konumu, gövde (FLU)
R_FLU_CAM = Rotation.from_euler("y", np.pi / 2).as_matrix()      # camera_link pozu: pitch 90° (aşağı bakar)
R_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # Adım C ile aynı
YAW_NED = math.pi / 2        # burun doğuya (ENU yaw 0) — PyBullet'teki gibi
MODEL_Z_REST = 0.013         # dron modeli yerde dururken model orijini ayak altının 0.013 m altında (Gazebo ölçümü)
PAD_TOP_W    = 0.001         # pad yüzeyi (dünya z)

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class LandingNode(Node):
    def __init__(self, a):
        super().__init__("landing_node")
        self.a = a
        self.pol = None if a.no_rl else PolicyNP(PROJ / "reports/faz7_res/policy_np.npz")
        self.det = make_detector("subpix")

        # PX4
        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_sp   = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.pub_cmd  = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_odom, PX4_QOS)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self.on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, "/fmu/out/vehicle_land_detected", self.on_land, PX4_QOS)
        # kamera (ros_gz_bridge)
        self.create_subscription(Image, "/camera/image", self.on_image, 1)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, 1)
        # gerçek konum (yalnız KAYIT/değerlendirme için; kontrolde kullanılmaz): ilk kayıt = dron modeli (ayak hizası)
        self.truth = None
        self.create_subscription(TFMessage, "/gz/poses", self.on_truth, 1)
        self.odom = self.status = self.land = self.K = None
        self.frame, self.frame_id = None, 0
        self.state, self.n = "BEKLE", 0
        self._warned = False
        self.pad_samples = []
        out = PROJ / "reports/faz8_d"; out.mkdir(parents=True, exist_ok=True)
        # Kayıt BELLEKTE tutulur, uçuş bitince yazılır: 30 Hz döngüde /mnt/c'ye (Windows diski) yazmak döngüyü
        # yavaşlatabilir; döngü 1 s'den uzun durursa PX4 "offboard sinyal kaybı" failsafe'ine geçer.
        self.log_path, self.rows = out / f"{a.tag}.csv", []
        self.rows.append(["t", "state", "phase", "hold", "meas", "x", "y", "z", "vx", "vy", "vz",
                           "px", "py", "pz", "tx", "ty", "tz", "a0", "a1", "a2",
                           "gx", "gy", "gz", "reset_z", "landed"])
        self.result = None
        self.create_timer(DT, self.tick)
        self.get_logger().info(f"basladi | RL {'KAPALI' if a.no_rl else 'ACIK'} | kayit: {out / (a.tag + '.csv')}")

    # ------------------------------------------------------------------ abonelikler
    def on_odom(self, m):   self.odom = m
    def on_status(self, m): self.status = m
    def on_land(self, m):   self.land = m
    def on_truth(self, m):
        tr = m.transforms[0].transform.translation
        self.truth = np.array([tr.x, tr.y, tr.z])
    def on_info(self, m):
        self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(m.d, dtype=np.float64) if len(m.d) else np.zeros(5)
    def on_image(self, m):
        self.frame, self.frame_id = m, self.frame_id + 1

    # ------------------------------------------------------------------ PX4 yardımcıları
    def now_us(self):
        # 0 = "PX4, mesajı aldığın anın saatini bas" (uxrce_dds_client böyle yapıyor). ROS duvar saati ile PX4'ün
        # simülasyon saati arasındaki eşitleme farkından hiç etkilenmez.
        return 0

    def send_cmd(self, command, p1=0.0, p2=0.0):
        c = VehicleCommand(timestamp=self.now_us(), command=command, param1=float(p1), param2=float(p2),
                           target_system=1, target_component=1, source_system=1, source_component=1,
                           from_external=True)
        self.pub_cmd.publish(c)

    def send_setpoint(self, p_enu_body):
        """p_enu_body: GÖVDE MERKEZİ hedefi (yerel ENU)."""
        self.pub_mode.publish(OffboardControlMode(timestamp=self.now_us(), position=True))
        sp = TrajectorySetpoint(timestamp=self.now_us())
        sp.position = [float(v) for v in ned_from_enu(p_enu_body)]
        sp.velocity = [math.nan] * 3
        sp.acceleration = [math.nan] * 3
        sp.yaw = YAW_NED
        self.pub_sp.publish(sp)

    def drone_state(self):
        """Yerel ENU: gövde konumu, hız, R (FLU->ENU), açısal hız (FRD) + PX4 biçimi (ObsBuilder için)."""
        o = self.odom
        p_ned, v_ned = np.array(o.position, float), np.array(o.velocity, float)
        q = np.array(o.q, float)
        R = R_enu_flu_from_R_ned_frd(R_from_q_wxyz(q))
        return enu_from_ned(p_ned), enu_from_ned(v_ned), R, q, np.array(o.angular_velocity, float)

    # ------------------------------------------------------------------ algı
    def measure_pad(self, p_body, R):
        """Son kareden pad merkezini (yerel ENU) çöz; yoksa None."""
        if self.frame is None or self.K is None:
            return None
        m = self.frame
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids = detect_tags(gray, self.det)
        if ids is None:
            return None
        M_cw = R @ R_FLU_CAM @ R_LINK_OPT
        res = solve_pad(corners, ids, self.K, self.dist, M_cw, p_body + R @ MOUNT)
        return None if res is None else res["z"]

    # ------------------------------------------------------------------ 30 Hz döngü
    def tick(self):
        now = time.monotonic()
        if getattr(self, "_t_last", None) is not None:
            gap = now - self._t_last
            self.max_gap = max(getattr(self, "max_gap", 0.0), gap)
            if gap > 0.3:
                self.get_logger().warn(f"DONMA {gap:.2f} s | durum={self.state} k={getattr(self, 'k', 0)}")
        self._t_last = now
        if self.odom is None or self.status is None:
            return
        self.n += 1
        p, v, R, q, w = self.drone_state()
        st = self.status
        armed = st.arming_state == VehicleStatus.ARMING_STATE_ARMED
        offb = st.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

        if self.state == "BEKLE":                       # >=1 s setpoint akıt, sonra offboard + arm
            if self.n == 1:                             # başlamadan önce: dron yerde ve disarm mı?
                if self.land is None:                   # PX4'ün "yerde misin" mesajı henüz gelmedi (açılışta normal)
                    self.n = 0; return                  # sessizce bekle
                why = "dron arm'li" if armed else ("dron havada" if not self.land.landed else None)
                if why is not None:                     # gerçekten beklemek gerekiyor
                    if not self._warned:                # uyarıyı yalnız BİR KEZ bas
                        self.get_logger().warn(f"{why} -> yere inip disarm olmasi bekleniyor")
                        self._warned = True
                    self.n = 0; return
                self.home = p.copy()                    # hazır: kalkış noktasını kaydet

            self.send_setpoint(self.home)
            if self.n > CTRL_HZ:
                if not offb:
                    self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                elif not armed:
                    self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                else:
                    self.state, self.t_state = "KALKIS", time.time()
                    self.get_logger().info("offboard + arm tamam -> KALKIS")
            return

        a = self.a
        start = np.array([a.pad_x + a.dx, a.pad_y + a.dy, self.home[2] + a.h])   # gövde hedefi (home = yerdeki gövde)
        if self.state == "KALKIS":                      # önce dikey çık, sonra başlangıç noktasına git
            tgt = np.array([self.home[0], self.home[1], start[2]]) if p[2] < start[2] - 0.15 else start
            self.send_setpoint(tgt)
            if self.n % CTRL_HZ == 0:
                self.get_logger().info(f"KALKIS: konum={np.round(p, 2)} hedef={np.round(tgt, 2)} |v|={np.linalg.norm(v):.2f}")
            if np.linalg.norm(p - start) < 0.15 and np.linalg.norm(v) < 0.2:
                self.state, self.t_state = "BASLANGIC", time.time()
                self.get_logger().info("baslangic noktasinda -> BASLANGIC (3 s, pad olcumu)")
            return

        if self.state == "BASLANGIC":                   # dur, pad'i ölç, orijini kur
            self.send_setpoint(start)
            if self.frame_id % 2 == 0:
                z = self.measure_pad(p, R)
                if z is not None:
                    self.pad_samples.append(z)
            if time.time() - self.t_state > 3.0:
                if len(self.pad_samples) < 10:
                    self.get_logger().error(f"pad yeterince gorulmedi ({len(self.pad_samples)} olcum) -> iptal")
                    return self.finish("PAD_YOK")
                pad0 = np.median(np.array(self.pad_samples), axis=0)
                self.origin = pad0 - np.array([0.0, 0.0, PLATFORM_TOP])
                self.get_logger().info(f"pad (yerel ENU) = {np.round(pad0, 3)} | orijin = {np.round(self.origin, 3)}")
                self.begin_landing()
            return

        if self.state in ("KALKIS", "BASLANGIC", "INIS") and not offb:   # PX4 kontrolü aldı (failsafe vb.)
            self.get_logger().error(f"PX4 offboard'dan cikti (nav_state={st.nav_state}) -> iptal")
            self.result = dict(ok=False, why="PX4_FAILSAFE")
            return self.finish("PX4_FAILSAFE")

        if self.state == "INIS":
            self.landing_step(p, v, R, q, w)
            return

        if self.state == "BITTI":
            if self.land is not None and self.land.landed and armed:
                self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
            if not armed or time.time() - self.t_state > 15.0:
                with open(self.log_path, "w", newline="") as fh:
                    csv.writer(fh).writerows(self.rows)
                raise SystemExit

    def begin_landing(self):
        self.imm = PlatformIMM(dt=DT, wave_hz=WAVE_HZ, meas_var=MEAS_STD ** 2)
        self.gate = DescentGate({})
        self.guid = LandingGuidance(t_land=T_LAND); self.guid.reset_schedule()
        self.ob = ObsBuilder(origin_enu=self.origin + np.array([0.0, 0.0, FOOT]))  # gövde -> ayak -> PyBullet
        self.steps_since_meas, self.k, self.last_fid = 0, 0, -1
        self.plat_pos_est = np.zeros(3); self.plat_vel_est = np.zeros(3)
        self.state, self.t_state = "INIS", time.time()
        self.get_logger().info("-> INIS")

    def landing_step(self, p, v, R, q, w):
        """LandingAviary.step ile aynı sıra: gözlem (kamera + IMM) -> politika -> kapı + güdüm -> setpoint."""
        self.k += 1
        pos = p - np.array([0.0, 0.0, FOOT]) - self.origin      # ayak noktası, PyBullet çerçevesi
        # 1) ölçüm (15 Hz, yeni kare varsa)
        z = None
        if self.k % (CTRL_HZ // CAM_HZ) == 0 and self.frame_id != self.last_fid:
            self.last_fid = self.frame_id
            zl = self.measure_pad(p, R)
            z = None if zl is None else zl - self.origin
        self.steps_since_meas = 0 if z is not None else self.steps_since_meas + 1
        xe = np.ravel(self.imm.step(z)[0]) if z is not None else np.ravel(self.imm.predict_only())
        self.plat_pos_est = np.array([xe[0], xe[2], xe[4]]); self.plat_vel_est = np.array([xe[1], xe[3], xe[5]])
        # 2) gözlem + politika
        if self.pol is not None:
            obs = self.ob.build(self.odom.position, self.odom.velocity, q, w,
                                self.plat_pos_est - pos, self.plat_vel_est - v)
            act = self.pol.act(obs)
        else:
            act = np.zeros(3)
        self.ob.push_action(act)
        # 3) kapı + güdüm + residual
        hold = self.gate.step(pos, v, self.plat_pos_est, self.plat_vel_est, self.steps_since_meas)
        base, phase = self.guid.position_command(pos, v, self.plat_pos_est, self.plat_vel_est, DT, hold=hold)
        target = base + RES_SCALE * act
        self.send_setpoint(target + self.origin + np.array([0.0, 0.0, FOOT]))
        t = self.k * DT
        self.rows.append([f"{t:.3f}", self.state, phase, int(hold), int(z is not None), *np.round(pos, 4),
                           *np.round(v, 4), *np.round(self.plat_pos_est, 4), *np.round(target, 4), *np.round(act, 4),
                           *(np.round(self.truth, 4) if self.truth is not None else [""] * 3),
                           self.odom.reset_counter, int(bool(self.land and self.land.landed))])
        # 4) bitiş — PyBullet env gibi GERÇEK durumla (env'in terminated/is_success'i de gerçek durumu kullanır):
        #    başarı = yatay < 0.3, h < 0.05, |v_rel| < 0.5. Kontrol zinciri gerçek konumu hiç görmez.
        if self.truth is None:
            return
        h = self.truth[2] + MODEL_Z_REST - PAD_TOP_W                      # ayak altı − pad üstü (m)
        horiz = np.linalg.norm(self.truth[:2] - np.array([self.a.pad_x, self.a.pad_y]))
        vrel = np.linalg.norm(v)                                           # pad sabit
        touched = h < 0.05 or (self.land is not None and self.land.landed)
        if touched or t > EP_LEN:
            ok = touched and horiz < 0.3 and vrel < 0.5
            self.result = dict(ok=ok, t=t, h=h, horiz=horiz, vrel=vrel)
            self.get_logger().info(f"BITTI: {'BASARILI' if ok else 'BASARISIZ'} | t={t:.1f} s h={h:.3f} "
                                   f"yatay={horiz:.3f} m |v_rel|={vrel:.2f} m/s | phase={phase}")
            self.finish("OK" if ok else "FAIL")

    def finish(self, why):
        self.send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.state, self.t_state = "BITTI", time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rl", action="store_true", help="residual = 0 (yalnız klasik yapı)")
    ap.add_argument("--tag", default="d_deneme")
    ap.add_argument("--pad-x", type=float, default=1.5, help="pad konumu, yerel ENU (dünya dosyasındaki pose)")
    ap.add_argument("--pad-y", type=float, default=0.0)
    ap.add_argument("--dx", type=float, default=0.3, help="başlangıç: pad'e göre yatay ofset (m)")
    ap.add_argument("--dy", type=float, default=-0.2)
    ap.add_argument("--h", type=float, default=0.8, help="başlangıç: ayakların yerden yüksekliği (m)")
    a = ap.parse_args()
    rclpy.init(); node = LandingNode(a)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    print("SONUC:", node.result, "| en uzun dongu boslugu:", round(getattr(node, "max_gap", 0.0), 3), "s")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()