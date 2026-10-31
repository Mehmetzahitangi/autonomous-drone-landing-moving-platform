import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p
from gymnasium import spaces
from estimation.imm_ekf import PlatformIMM
from guidance.landing_guidance import LandingGuidance

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
                 act: ActionType=ActionType.RPM):
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

        """

        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8

        self.plat_drift = np.array([0.1, 0.0])  # m/s  (x,y sürüklenme)
        self.plat_amp   = 0.2                    # m    (dalga genliği)
        self.plat_freq  = 0.3                    # Hz   (dalga frekansı)

        self.ACT_SCALE = 0.1   # adım başına +- 0.1 m hedef offset → 9° eğim (nazik). Yavaş kalırsa 0.2'ye çıkar.

        self.meas_std = 0.03        # sensör ölçüm gürültüsü std(m), ArUco/görsel deki gibi
        self.use_estimator = True   # True: obs'de kestirim kullanır, gerçekçi. False: eski ground-truth A/B testi için kullanılabilir

        self.RES_SCALE = 0.05
        self.use_residual = True

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

    def _relative_state_est(self):

        s = self._getDroneStateVector(0)
        drone_pos, drone_vel = s[0:3], s[10:13]
        t = self.step_counter / self.PYB_FREQ

        # gerçek platform (x,y) + ölçüm gürültüsü = sensör ölçümü
        plat_x = self.plat_drift[0] * t
        plat_y = self.plat_amp * np.sin(2*np.pi*self.plat_freq*t)

        z = np.array([plat_x, plat_y]) + np.random.normal(0, self.meas_std, 2)

        xe = np.ravel(self._imm.step(z)[0])          # [px, vx, py, vy]

        #x_est, _ = self._imm.step(z)                       # [px, vx, py, vy]
        #x_est = np.ravel(x_est)                            # filterpy (4,1) kolon vektörü → (4,)
        self.plat_pos_est = np.array([xe[0], xe[2], self.PLATFORM_TOP]) # mutlak kestirimi self'e yazıyoruz, _preprocessAction tabanı bunu okuyacak
        self.plat_vel_est = np.array([xe[1], xe[3], 0.0])
        return self.plat_pos_est - drone_pos, self.plat_vel_est - drone_vel

    def reset(self, seed=None, options=None):

        self._imm  = PlatformIMM(dt=self.CTRL_TIMESTEP, wave_hz=self.plat_freq, meas_var=self.meas_std**2)
        self._guid = LandingGuidance(t_land=self.EPISODE_LEN_SEC) #  klasik ZEM/ZEV tabanı

        self._guid.reset_schedule() # # her bölüm: iç zaman + alçalma çizelgesini sıfırla
        
        x0, y0 = np.random.uniform(-0.5, 0.5, size=2)
        z0 = np.random.uniform(0.6, 1.0)
        self.INIT_XYZS = np.array([[x0, y0, z0]])   
        

        # IMM-EKF  (dt = kontrol periyodu)
        self._imm = PlatformIMM(dt=self.CTRL_TIMESTEP, wave_hz=self.plat_freq, meas_var=self.meas_std**2)
    
        obs = super().reset(seed=seed, options=options)
        r_rel, _ = self._relative_state()       # prev_dist GERÇEK durumdan
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
        # İnce iniş pedi (1 m'lik küp yerine düz bir plaka). baseMass=0 -> statik,
        # zaten her adım resetBasePositionAndOrientation ile taşıyoruz.
        half = [0.30, 0.30, 0.02]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[0.2, 0.6, 1.0, 1.0], physicsClientId=self.CLIENT)
        self.PLATFORM_ID = p.createMultiBody(baseMass=0.0,
                                             baseCollisionShapeIndex=col,
                                             baseVisualShapeIndex=vis,
                                             basePosition=[0, 0, 0.05],
                                             physicsClientId=self.CLIENT)
        self.PLATFORM_TOP = 0.05 + half[2]   # pedin üst yüzeyi (yaklaşık 0.07 m yükseklik)

    def _move_platform(self):
        t = self.step_counter / self.PYB_FREQ            # saniye cinsinden sim zamanı
        x = self.plat_drift[0] * t
        y = self.plat_amp * np.sin(2*np.pi*self.plat_freq*t)   # dalga
        p.resetBasePositionAndOrientation(self.PLATFORM_ID,
                                        [x, y, 0.05],
                                        p.getQuaternionFromEuler([0, 0, 0]),
                                        physicsClientId=self.CLIENT)
    
    def _computeInfo(self):
        r_rel, v_rel = self._relative_state()

        h = max(0, -r_rel[2])

        horiz = np.linalg.norm(r_rel[0:2])
        vrel  = np.linalg.norm(v_rel)

        success = bool(horiz < 0.3 and h < 0.05 and vrel < 0.5)   # terminated'daki başarı koşuluyla aynı

        return {"is_success": success, "touchdown_speed": float(vrel), "horiz": float(horiz)}

    def _relative_state(self):
        s = self._getDroneStateVector(0)
        drone_pos, drone_vel = s[0:3], s[10:13]
        t = self.step_counter / self.PYB_FREQ
        plat_pos = np.array([self.plat_drift[0]*t,
                            self.plat_amp*np.sin(2*np.pi*self.plat_freq*t),
                            self.PLATFORM_TOP])
        plat_vel = np.array([self.plat_drift[0],
                            self.plat_amp*2*np.pi*self.plat_freq*np.cos(2*np.pi*self.plat_freq*t),
                            0.0])
        return plat_pos - drone_pos, plat_vel - drone_vel   # r_rel, v_rel

    def _observationSpace(self):
        """Gözlem uzayını [r_rel(3), v_rel(3)] içerecek şekilde genişletir."""

        base_space = super()._observationSpace()
        
        if self.OBS_TYPE == ObservationType.KIN:
            # 6 yeni değişken (3 konum + 3 hız) için sonsuz (-inf, +inf) sınırlar oluştur
            extra_low = np.full((self.NUM_DRONES, 6), -np.inf, dtype=np.float32)
            extra_high = np.full((self.NUM_DRONES, 6), np.inf, dtype=np.float32)
            
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
            
            # Her dron için ek gözlemleri dizi formatına çevirir
            extra_obs = np.array([np.hstack([r_rel, v_rel]) for _ in range(self.NUM_DRONES)])
            
            # Temel gözlemlerle yan yana birleştirilir
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

            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=dt,
                cur_pos=drone_pos,  cur_quat=state[3:7],
                cur_vel=drone_vel,  cur_ang_vel=state[13:16],
                target_pos=target_pos)
            rpm[k, :] = rpm_k
        return rpm
    
            #target_pos = state[0:3] + self.ACT_SCALE * action[k, :]   # # SAC tüm hedefi veriyor
            #Residual RL:
            #target_base, phase = self._guid.position_command(drone_pos, drone_vel, plat_pos_est, plat_vel_est, dt)
            #target_pos = target_base + self.RES_SCALE * action[k, :]  # SAC sadece düzeltir


            #rpm_k, _, _ = self.ctrl[k].computeControl(
            #    control_timestep=self.CTRL_TIMESTEP,
            #    cur_pos=state[0:3],  cur_quat=state[3:7],
            #    cur_vel=state[10:13], cur_ang_vel=state[13:16],
            #    target_pos=target_pos)
            #rpm[k, :] = rpm_k
        #return rpm"""