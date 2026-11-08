import pathlib

import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p
from gymnasium import spaces
from estimation.imm_ekf import PlatformIMM
from guidance.landing_guidance import LandingGuidance
from safety.cbf_shield import VerticalContactCBF
from perception.aruco_detector import ArucoCamera, cam_config

class LandingAviary(BaseRLAviary):
    """LandingAviary

    Landing task for a single drone.

    The goal is to land the drone on a platform.

    The episode is considered successful if the drone lands on the platform and stays there for a certain amount of time.

    The episode is considered failed if the drone crashes or flies away from the platform.

    The episode ends when the drone lands on the platform, crashes, flies away from the platform, or when the maximum number of steps is reached.
    """


    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int=240,
                 ctrl_freq: int=30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 use_cbf: bool=False,
                 dr_wind: bool=False,
                use_aruco: bool=False,
                cam_cfg=None,):
        """Initialization of a LandingAviary object.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (model), by default DroneModel.CF2X
        initial_xyzs : list, optional
            List of lists of initial XYZ positions for each drone, by default None
        initial_rpys : list, optional
            List of lists of initial RPY orientations for each drone, by default None
        physics : Physics, optional
            The desired physics updates implementation, by default Physics.PYB
        pyb_freq : int, optional
            The frequency (Hz) at which the physics engine runs, by default 240
        ctrl_freq : int, optional
            The frequency (Hz) at which the control loop runs, by default 30
        gui : bool, optional
            Whether to use PyBullet's GUI, by default False
        record : bool, optional
            Whether to record a video of the simulation (only available with GUI), by default False
        obs : ObservationType, optional
            The desired observation type (sensor suite), by default ObservationType.KIN
        act : ActionType, optional
            The desired action type (control suite), by default ActionType.RPM
        dr_wind : bool, optional
            Whether to use wind disturbances, by default False
        """

        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8

        self.plat_drift = np.array([0.1, 0.0])  # m/s  (x,y sürüklenme)
        self.plat_amp   = 0.2                    # m    (dalga genliği)
        self.plat_freq  = 0.3                    # Hz   (dalga frekansı)

        self.ACT_SCALE = 0.1   # adım başına +- 0.1 m hedef offset → 9 derece eğim (nazik). Yavaş kalırsa 0.2'ye çıkar.

        self.meas_std = 0.03        # sensör ölçüm gürültüsü std(m), ArUco/görsel deki gibi
        self.use_estimator = True   # True: obs'de kestirim kullanır, gerçekçi. False: eski ground-truth A/B testi için kullanılabilir

        self.RES_SCALE = 0.05
        self.use_residual = True

        self.plat_phase=0.0
        self.plat_x0=0.0
        self.plat_y0=0.0

        self.heave_amp=0.0
        self.heave_phase=0.0

        self.use_cbf = use_cbf
        self._cbf = None

        self.WIND_MAX = 0.03          # N, DR yatay rüzgar üst sınırı
        self._wind = np.zeros(3)
        self.dr_wind = dr_wind        # True: reset'te rastgele sabit rüzgar örnekle

        self.MARKER_ID   = 23
        self.MARKER_SIZE = 0.42   # m, fiziksel marker kenarı (make_marker.py layout'u ile eşleşir)

        self.use_aruco = use_aruco
        # cam_cfg: None/'ideal' (eski davranış) | 'realistic' | dict (bkz. perception/aruco_detector.py CAM_PRESETS)
        self.cam_cfg = cam_config(cam_cfg)
        self._cam = None
        self._marker_visible = False
        self._meas_queue = []         # (teslim_tik, z_dunya, yakalama_tik) — gecikmeli ölçümler
        self._tick = 0                # kontrol-adımı sayacı (yalnız algı zamanlaması için)
        self._cam_frames = 0          # bölümde yakalanan kare sayısı
        self._cam_dets = 0            # bölümde tag bulunan kare sayısı
        self._meas_age = 0            # son teslim edilen ölçümün yaşı (adım)
        self._sens_rng = np.random.default_rng()

        self.init_xy_max = 0.5            # yatay baslangic ofset siniri (curriculum ile ayarlanir)
        self.init_z_lo, self.init_z_hi = 0.6, 1.0

        self._steps_since_meas = 0    # son olcumden kac adim gecti

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)




    def _computeReward(self):
        """Ödül fonksiyonu: Hizalanma, yaklaşma, yumuşak temas ve düzgünlük cezaları/bonusları."""

        s = self._getDroneStateVector(0)
        r_rel, v_rel = self._relative_state()

        curr_dist = np.linalg.norm(r_rel)
        horiz_dist = np.linalg.norm(r_rel[0:2])
        # r_rel = plat_pos - drone_pos olduğu için h = drone_pos.z - plat_pos.z = -r_rel[2]
        h = max(0, -r_rel[2]) 
        vel_mag = np.linalg.norm(v_rel)

        # Hiperparametreler (Bunlar Faz 3 ölçümlerine göre ayarlanacak)
        # Dron asılı kalırsa = ödül hackleme → adım cezasını artır, kendini çakıyorsa → z/temas dengesini ayarla
        w1, w2, w3, w4, w5 = 10.0, 1.0, 1.0, 3.0, 0.05
        step_penalty = 0.05
        bonus_success = 150.0
        penalty_crash = 50.0

        # 1. Yaklaşma (Potansiyel-tabanlı şekillendirme)
        # Dron platforma her yaklaştığında pozitif, uzaklaştığında negatif ödül alır
        reward_approach = w1 * (self.prev_dist - curr_dist)

        # 2. & 3. Hizalanma ve Alçalma (Mesafe cezaları)
        penalty_xy = -w2 * horiz_dist
        penalty_z = -w3 * h

        # 4. Yumuşak Temas (Proximity ile çarpılmış)
        # Dron yukarıdayken ceza çok azdır, pedin tam üstüne indiğinde maksimum hız cezası uygulanır.
        # Yüksekteyken hızlı uçamaya izin var ama platforma yaklaşırken yavaşlamasını gerektiğini bu şekilde ödül/ceza ile değerlendiriyoruz.
        proximity = np.exp(-3.0 * h) 
        penalty_vel = -w4 * vel_mag * proximity

        # 5. Düzgünlük (Aksiyon büyüklüğü cezası)
        # "Hedefe git ama bunu yaparken kumandaları olabildiğince az ve yumuşak kullan. Gereksiz agresif manevralar yapma." Bu, minimum enerji (min-effort) prensibidir.
        # Sürekli aksiyonlar yerine daha yumuşak ve küçük aksiyonlar ile hedefe ulaşmak ödüllendirilir. Titreme ve agresif manevralar cezalandırılır.
        last_action = self.action_buffer[-1][0] if len(self.action_buffer) > 0 else np.zeros(4)
        penalty_action = -w5 * np.linalg.norm(last_action)**2

        # Toplam adım ödülü
        step_reward = reward_approach + penalty_xy + penalty_z + penalty_vel + penalty_action - step_penalty

        # Aktif algi tesviki: marker gorunurse kucuk bonus -> markeri kadrajda tutmayi ogrenir
        if self.use_aruco:
            step_reward += 0.3 * float(self._marker_visible)

        #  Terminal Bonus ve Cezaları 

        # Platforma temas edildi (z ekseninde pedin üstüne ve yatayda ped sınırlarına girildi)
        if horiz_dist < 0.3 and h < 0.05:
            if vel_mag < 0.5: # Yumuşak iniş eşiği
                step_reward += bonus_success
            else:
                step_reward -= penalty_crash # Yüksek hızla çakılma

        # Dron platformu ıskalayıp yere çakıldıysa veya takla atıyorsa
        elif s[2] < self.PLATFORM_TOP - 0.05:   # yalnız YERE çakılma; eğim cezalandırılmıyor
            step_reward -= penalty_crash

        # Sonraki adım için mesafeyi güncelle
        self.prev_dist = curr_dist
        
        return step_reward


    def _computeTerminated(self):
        """Başarı (yumuşak iniş) veya Başarısızlık (çakılma/devrilme) durumları."""

        s = self._getDroneStateVector(0)
        r_rel, v_rel = self._relative_state()
        
        h = max(0, -r_rel[2])
        horiz_dist = np.linalg.norm(r_rel[0:2])
        vel_mag = np.linalg.norm(v_rel)
        
        # Başarı: Platformun üstünde ve yavaş
        if horiz_dist < 0.3 and h < 0.05 and vel_mag < 0.5:
            return True
            
        # Başarısızlık 1: Platforma hızlı çarpma (crash)
        if horiz_dist < 0.3 and h < 0.05 and vel_mag >= 0.5:
            return True
            
        # Başarısızlık 2: Platformu ıskalayıp yer seviyesine inme
        if s[2] < self.PLATFORM_TOP - 0.05:
            return True
            
        # Başarısızlık 3: Aşırı eğilme / Takla atma tehlikesi
        # 7. ve 8. indeksler dronun Euler açılarını (radyan cinsinden) temsil eder
        # s[7] = Roll (Yatma/Yuvarlanma): Dronun sağa veya sola yatıklığı (X ekseni etrafında dönüş).
        # s[8] = Pitch (Yunuslama/Şahlanma): Dronun öne veya arkaya yatıklığı (Y ekseni etrafında dönüş).
        # s[9] ise Yaw yani kendi etrafında dönme açısıdır)

        # 0.4 radyan yaklaşık 22.9 derece. Bu, dronun ciddi şekilde eğildiğini ve kontrolünü kaybetme riskinin yüksek olduğunu gösterir.
        # YENİ: sadece gerçek devrilme (kurtarılamaz), normal eğim artık öldürmez
        if abs(s[7]) > 1.2 or abs(s[8]) > 1.2:   # 1.2 yaklaşık 70 derece. Bu, dronun neredeyse yan yatmış olduğunu ve kontrolünü kaybettiğini gösterir.
            return True
            
        return False
        
    
    def _computeTruncated(self):
        """Zaman aşımı (Timeout) veya eğitim sınırları dışına çıkma."""

        s = self._getDroneStateVector(0)
        
        # Zaman aşımı
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
            
        # Dron arena sınırlarından fazla uzaklaştı
        if abs(s[0]) > 2.0 or abs(s[1]) > 2.0 or s[2] > 3.0:
            return True
            
        return False

    def _platform_state(self, t):
        # Platformun gerçek konum ve hızını tek bir yere toplayıp, döndürür 
        w = 2*np.pi*self.plat_freq

        x = self.plat_x0 + self.plat_drift[0]*t
        y = self.plat_y0 + self.plat_drift[1]*t + self.plat_amp*np.sin(w*t + self.plat_phase)
        z  = self.PLATFORM_TOP + self.heave_amp*np.sin(w*t + self.heave_phase)

        vz = self.heave_amp*w*np.cos(w*t + self.heave_phase)
        pos = np.array([x, y, z])
        vel = np.array([self.plat_drift[0],
                        self.plat_drift[1] + self.plat_amp*w*np.cos(w*t + self.plat_phase),
                        vz])
        return pos, vel

    def _relative_state_est(self):
        s = self._getDroneStateVector(0)
        drone_pos, drone_vel = s[0:3], s[10:13]
        t = self.step_counter / self.PYB_FREQ

        if self.use_aruco:
            z = self._camera_pipeline()                  # bu adımda TESLİM edilen ölçüm ya da None
            self._marker_visible = (z is not None)
            self._steps_since_meas = 0 if z is not None else self._steps_since_meas + 1
            if z is not None:
                xe = np.ravel(self._imm.step(z)[0])      # ölçümle güncelle
            else:
                xe = np.ravel(self._imm.predict_only())  # ölçüm yok (kare yok / tag yok / yolda) → tahmin
        else:
            # eski yol: gerçek platform + Gauss gürültü (fallback)
            plat_pos, _ = self._platform_state(t)
            z = plat_pos + np.random.normal(0, self.meas_std, 3)
            xe = np.ravel(self._imm.step(z)[0])

        self.plat_pos_est = np.array([xe[0], xe[2], xe[4]])
        self.plat_vel_est = np.array([xe[1], xe[3], xe[5]])
        return self.plat_pos_est - drone_pos, self.plat_vel_est - drone_vel

    def _camera_pipeline(self):
        """Gerçekçi algı zamanlaması: kamera kendi hızında kare yakalar (cfg['hz']), her ölçüm
        işleme+iletim gecikmesiyle (latency ± jitter) kuyruğa girer, vakti gelince teslim edilir.
        Teslim edilen ölçüm, YAKALANDIĞI andaki pad konumunu taşır (yani bayattır).
        ideal cfg (hz=None, latency=0) → her adım kare, anında teslim = eski davranış."""
        c = self.cam_cfg
        if self._cam is None:   # ilk kullanımda (DRONE_IDS hazırken) kur
            self._cam = ArucoCamera(self.CLIENT, self.DRONE_IDS[0], cfg=c, rng=self._sens_rng)

        period = 1 if not c["hz"] else max(1, int(round(self.CTRL_FREQ / c["hz"])))
        if self._tick % period == 0:
            m = self._cam.measure()
            self._cam_frames += 1
            if m is not None:
                self._cam_dets += 1
                lat = c["latency_s"] + self._sens_rng.uniform(-1, 1) * c["latency_jitter_s"]
                delay = max(0, int(round(lat * self.CTRL_FREQ)))
                self._meas_queue.append((self._tick + delay, m["z"], self._tick))

        ready = [q for q in self._meas_queue if q[0] <= self._tick]
        self._meas_queue = [q for q in self._meas_queue if q[0] > self._tick]
        z = None
        if ready:
            _, z, cap = max(ready, key=lambda q: q[2])   # en taze yakalanan
            self._meas_age = self._tick - cap
        self._tick += 1
        return z

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)                 # motor itkileri (orijinal)
        if self._wind is not None and np.any(self._wind):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                                     physicsClientId=self.CLIENT)
            p.applyExternalForce(self.DRONE_IDS[nth_drone], -1,
                                 forceObj=self._wind.tolist(), posObj=pos,
                                 flags=p.WORLD_FRAME, physicsClientId=self.CLIENT)
    

    def set_wind(self, a_xy):      # a_xy ∈ [-1,1]^2 (RARL adversary aksiyonu); WIND_MAX ile ölçeklenir
        self._wind = np.array([self.WIND_MAX*float(a_xy[0]), self.WIND_MAX*float(a_xy[1]), 0.0])

    def set_init_difficulty(self, xy_max=None, z_lo=None, z_hi=None):
        """Curriculum: baslangic zorlugunu ayarla (kucuk ofset = kolay)."""
        if xy_max is not None: self.init_xy_max = float(xy_max)
        if z_lo   is not None: self.init_z_lo   = float(z_lo)
        if z_hi   is not None: self.init_z_hi   = float(z_hi)

    def reset(self, seed=None, options=None):
                
        self._cam = None   # her reset'te DRONE_IDS yenilenir, kamerayı yeniden bağlar
        self._steps_since_meas = 0
        self._meas_queue, self._tick = [], 0
        self._cam_frames = self._cam_dets = self._meas_age = 0

        # domain randomization: her bölümde rastgele sabit rüzgar (yön + şiddet)
        if self.dr_wind:
            wang = np.random.uniform(0, 2*np.pi)
            wmag = np.random.uniform(0.0, self.WIND_MAX)   # [0, WIND_MAX] N
            self._wind = np.array([wmag*np.cos(wang), wmag*np.sin(wang), 0.0])
        else:
            self._wind = np.zeros(3)                        # eval/diag: rüzgarsız

        # platform randomizationı (her bölüm farklı yörünge) 
        self.plat_phase = np.random.uniform(0, 2*np.pi) # Platform artık hep y=0'dan yukarı değil, farklı noktadan sallanmaya başlar
        self.plat_amp   = np.random.uniform(0.10, 0.30) # salınım genliği (m). Ne kadar geniş sallanıyor (deniz durumu şiddeti)
        self.plat_freq  = np.random.uniform(0.25, 0.35) # salınım frekansı (Hz). Ne kadar hızlı sallanıyor (deniz durumu periyodu)
        ang = np.random.uniform(0, 2*np.pi); spd = np.random.uniform(0.0, 0.12)
        self.plat_drift = np.array([spd*np.cos(ang), spd*np.sin(ang)]) # sabit sürüklenme hızı [vx,vy]. Platformun düz kayması (seyreden gemi). ang+spd ile rastgele yöne.
        self.plat_x0, self.plat_y0 = 0.0, 0.0

        self.heave_amp   = np.random.uniform(0.0, 0.06)   # dikey genlik (m); 0 = heave yok
        self.heave_phase = np.random.uniform(0, 2*np.pi)

        # drone başlangıcı 
        x0, y0 = np.random.uniform(-self.init_xy_max, self.init_xy_max, size=2)
        z0 = np.random.uniform(self.init_z_lo, self.init_z_hi)
        self.INIT_XYZS = np.array([[x0, y0, z0]])

        # algı gürültüsü için ayrı RNG (senaryo çekilişlerinden SONRA → aynı seed = aynı senaryo,
        # kamera ayarı ne olursa olsun; ideal/gerçekçi karşılaştırması eşli kalır)
        self._sens_rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

        # IMM (randomize edilmiş freq ile) + klasik taban, IMM artık doğru (rastgele) freq'i alıyor.
        self._imm  = PlatformIMM(dt=self.CTRL_TIMESTEP, wave_hz=self.plat_freq, meas_var=self.meas_std**2)
        self._guid = LandingGuidance(t_land=self.EPISODE_LEN_SEC)
        self._guid.reset_schedule()
        self._cbf = VerticalContactCBF(look=self._guid.lookahead, dt=self.CTRL_TIMESTEP)

        # PID iç durumunu (integral + önceki hata) sıfırla. gym-pybullet-drones bunu yalnız
        # kurucuda yapıyor; yapılmazsa bir bölümün birikmiş integrali (konum ±2, tutum ±1)
        # sonraki bölüme taşınır → aynı seed sıralı koşuda farklı (bazen uzaklaşan) sonuç verir.
        for c in getattr(self, "ctrl", []):
            c.reset()

        obs = super().reset(seed=seed, options=options)
        r_rel, _ = self._relative_state()
        self.prev_dist = np.linalg.norm(r_rel)
        return obs
    
    def step(self, action):
        """Performs an environment step.

        Parameters
        ----------
        action : np.array
            The current action.

        Returns
        -------
        np.array
            The current observation of the environment.
        float
            The current reward.
        bool
            Whether the current episode is done.
        bool
            Whether the current episode timed out.
        dict
            A dictionary containing additional information.

        """
        self._move_platform()
        return super().step(action)

    def _addObstacles(self):
        # İnce iniş pedi (collision + görsel kutu). Marker AYRI bir quad olarak üstüne konur.
        half = [0.30, 0.30, 0.02]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                rgbaColor=[0.2, 0.6, 1.0, 1.0], physicsClientId=self.CLIENT)
        self.PLATFORM_ID = p.createMultiBody(baseMass=0.0,
                                            baseCollisionShapeIndex=col,
                                            baseVisualShapeIndex=vis,
                                            basePosition=[0, 0, 0.05],
                                            physicsClientId=self.CLIENT)
        self.PLATFORM_TOP = 0.05 + half[2]   # ~0.07 m

        # ArUco marker: ayrı ince QUAD mesh (doku kutuya düzgün oturmadığı için)
        base = pathlib.Path(__file__).resolve().parent.parent / "assets"
        mvis = p.createVisualShape(p.GEOM_MESH, fileName=str(base / "marker_quad.obj"),
                                rgbaColor=[1, 1, 1, 1], physicsClientId=self.CLIENT)
        self.MARKER_BODY = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                            baseVisualShapeIndex=mvis,
                                            basePosition=[0, 0, self.PLATFORM_TOP + 0.001],
                                            physicsClientId=self.CLIENT)
        tex = p.loadTexture(str(base / "aruco_marker.png"), physicsClientId=self.CLIENT)
        p.changeVisualShape(self.MARKER_BODY, -1, textureUniqueId=tex, physicsClientId=self.CLIENT)

    def _move_platform(self):
        t = self.step_counter / self.PYB_FREQ
        pos, _ = self._platform_state(t)
        base_z = pos[2] - 0.02   # pad yarı-kalınlığı
        p.resetBasePositionAndOrientation(self.PLATFORM_ID, [pos[0], pos[1], base_z],
                                            p.getQuaternionFromEuler([0, 0, 0]),
                                            physicsClientId=self.CLIENT)
        # marker quad'ı platform üst yüzeyinde tut (heave dahil)
        if hasattr(self, "MARKER_BODY"):
            p.resetBasePositionAndOrientation(self.MARKER_BODY, [pos[0], pos[1], pos[2] + 0.001],
                                                p.getQuaternionFromEuler([0, 0, 0]),
                                                physicsClientId=self.CLIENT)

    def _landing_supervisor(self, drone_pos, target_pos):
        """Durum-makinesi: marker gorunurluk + hizaya gore DIKEY inisi kapiya alir.
        gorunurluk+hizali -> normal in; hizasiz -> yukseklik tut; kayip -> biraz yuksel (yeniden yakala)."""
        tp = np.array(target_pos, dtype=float)
        horiz = np.linalg.norm(self.plat_pos_est[0:2] - drone_pos[0:2])
        ALIGN_TOL, LOST_MAX = 0.20, 5
        if self._marker_visible and horiz > ALIGN_TOL:
            tp[2] = drone_pos[2]                       # HIZALAN: yukseklik tut, yatay yaklas
        elif (not self._marker_visible) and self._steps_since_meas > LOST_MAX:
            tp[2] = min(drone_pos[2] + 0.05, 1.0)      # ARA / YENIDEN-YAKALA: biraz yuksel
        # else: ALCAL -> tp aynen (normal inis)
        return tp
    
    def _computeInfo(self):
        r_rel, v_rel = self._relative_state()

        h = max(0, -r_rel[2])

        horiz = np.linalg.norm(r_rel[0:2])
        vrel  = np.linalg.norm(v_rel)

        success = bool(horiz < 0.3 and h < 0.05 and vrel < 0.5)   # terminated'daki başarı koşuluyla aynı

        info = {"is_success": success, "touchdown_speed": float(vrel), "horiz": float(horiz),
                "vz_rel": float(v_rel[2]), "vxy_rel": float(np.linalg.norm(v_rel[0:2])), "h": float(h)}
        if hasattr(self, "plat_pos_est"):
            t = self.step_counter / self.PYB_FREQ
            info["est_err"] = float(np.linalg.norm(self.plat_pos_est - self._platform_state(t)[0]))
        if self.use_aruco:
            info.update(cam_frames=self._cam_frames, cam_dets=self._cam_dets,
                        stale=self._steps_since_meas, meas_age=self._meas_age)
        return info

    def _relative_state(self):
        s = self._getDroneStateVector(0)
        drone_pos, drone_vel = s[0:3], s[10:13]
        t = self.step_counter / self.PYB_FREQ
        plat_pos, plat_vel = self._platform_state(t) 
        return plat_pos - drone_pos, plat_vel - drone_vel   # r_rel, v_rel

    def _observationSpace(self):
        """Gözlem uzayını [r_rel(3), v_rel(3)] içerecek şekilde genişletir."""

        base_space = super()._observationSpace()
        
        if self.OBS_TYPE == ObservationType.KIN:
            # 6 yeni değişken (3 konum + 3 hız) için sonsuz (-inf, +inf) sınırlar oluştur
            n_extra = 6
            extra_low = np.full((self.NUM_DRONES, n_extra), -np.inf, dtype=np.float32)
            extra_high = np.full((self.NUM_DRONES, n_extra), np.inf, dtype=np.float32)

            
            # BaseRLAviary'den gelen uzayın sonuna ekle
            new_low = np.hstack([base_space.low, extra_low])
            new_high = np.hstack([base_space.high, extra_high])
            
            
            return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
            
        return base_space

    def _computeObs(self):
        """Orijinal gözlem dizisinin sonuna göreli durumu ekler."""
        base_obs = super()._computeObs()
        
        if self.OBS_TYPE == ObservationType.KIN:
            r_rel, v_rel = self._relative_state_est() if self.use_estimator else self._relative_state()
            feat = list(r_rel) + list(v_rel)
            extra_obs = np.array([feat for _ in range(self.NUM_DRONES)])
            new_obs = np.hstack([base_obs, extra_obs]).astype(np.float32)
            return new_obs
        return base_obs

    def _preprocessAction(self, action):
        """Aksiyonu MUTLAK 1m-sıçrama yerine mevcut konuma GÖRELİ küçük offset olarak uygula.
        Böylece DSL PID nazik eğim komutu üretir, dron takla atmaz. (bkz. N1)"""

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        dt = self.CTRL_TIMESTEP
        
        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            drone_pos, drone_vel = state[0:3], state[10:13]

            if self.use_residual:
                # Klasik ZEM/ZEV tabanı — gözlemle aynı kestirimi okur (self._plat_*_est).
                # Bu cache _relative_state_est her adım günceller, filtreyi BURADA tekrar adımlamıyoruz (yoksa adım başına iki kez adımlanırdı!!!)

                target_base, phase = self._guid.position_command(drone_pos, drone_vel, self.plat_pos_est, self.plat_vel_est, dt)
                target_pos = target_base + self.RES_SCALE * action[k, :]   # taban + residual, SAC sadece düzeltir

                # landing_diag da residual katkısını ölçmek için
                self._dbg_phase    = phase
                self._dbg_base_off = float(np.linalg.norm(target_base - drone_pos))
                self._dbg_res_off  = float(self.RES_SCALE * np.linalg.norm(action[k, :]))
            else:
                target_pos = drone_pos + self.ACT_SCALE * action[k, :]     # eski düz SAC yolu
                if self.use_aruco:                         # durum-makinesi sadece gorsel modda
                    target_pos = self._landing_supervisor(drone_pos, target_pos)

            if self.use_cbf:
                target_pos, self._dbg_cbf = self._cbf.filter(drone_pos, drone_vel, self.plat_pos_est, self.plat_vel_est, target_pos)
    
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=dt,
                cur_pos=drone_pos,  cur_quat=state[3:7],
                cur_vel=drone_vel,  cur_ang_vel=state[13:16],
                target_pos=target_pos)
            rpm[k, :] = rpm_k
        return rpm
    