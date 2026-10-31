"""
landing_guidance.py — Hareketli platforma yumuşak iniş için klasik güdüm (RL YOK).

İKİ FAZLI (planlanan yapı):
  * YAKLAŞMA (uzakta): ZEM/ZEV optimal güdüm yasası. Hem göreli konumu (ZEM) hem göreli
    hızı (ZEV) temas anında sıfıra sürer:  a_cmd = 6/t_go²·r_rel + 4/t_go·v_rel
    (r_rel = platform - dron, v_rel = platform - dron hızı; işaretler buna göre artı).
  * TERMINAL (yakında): kararlı takip + kademeli yumuşak alçalma (çarpmadan otur).

ZEM/ZEV bir İVME yasasıdır. Kararlı kalmak için çıktıyı kısa-ufuklu bir KONUM waypoint'ine
çeviriyoruz:  next_pos = p + v·look + 0.5·a·look²  ve ortamın PID aksiyonuyla sürüyoruz.
(VEL/hız aksiyonu bu kurulumda kararsızdı; konum/waypoint arayüzü kararlı.)

Girdi gerçek platform durumu; Faz 4'te yerine IMM-EKF kestirimi gelecek.
"""

import numpy as np


class LandingGuidance:
    def __init__(self,
                 t_land=6.0,          # YAKLAŞMA için ayrılan süre (t_go çizelgesi bundan)
                 t_go_min=0.4,        # t_go tabanı -> 1/t_go² patlamasını önler
                 a_max=3.0,           # ivme komutu doygunluğu (m/s²)
                 lookahead=0.4,       # ivme -> konum waypoint ufku (s)
                 h_switch=0.15,       # bu irtifanın (m) altında TERMINAL faza geç
                 descent_step=0.004,  # terminal fazda adım başı iniş (m) ~0.12 m/s @30Hz
                 contact_offset=0.05, # ped üstünde hedeflenecek pay (m)
                 **_):
        self.t_land = t_land
        self.t_go_min = t_go_min
        self.a_max = a_max
        self.lookahead = lookahead
        self.h_switch = h_switch
        self.descent_step = descent_step
        self.contact_offset = contact_offset
        self._t = 0.0
        self._cmd_z = None

    def reset_schedule(self):
        """Yeni bölüm başında zamanı ve alçalma çizelgesini sıfırla."""
        self._t = 0.0
        self._cmd_z = None

    def position_command(self, drone_pos, drone_vel, platform_pos, platform_vel, dt):
        """İstenen HEDEF KONUMU (3,) ve aktif fazı döndürür."""
        self._t += dt
        h = drone_pos[2] - platform_pos[2]
        contact_z = platform_pos[2] + self.contact_offset

        if h > self.h_switch:
            # --- YAKLAŞMA: ZEM/ZEV optimal yasası ---
            t_go = max(self.t_land - self._t, self.t_go_min)   # temasa kalan süre (çizelge)
            r_rel = platform_pos - drone_pos                    # istenen 0 (pede otur)
            v_rel = platform_vel - drone_vel                    # istenen 0 (hız eşle)
            a_cmd = (6.0 / t_go**2) * r_rel + (4.0 / t_go) * v_rel
            n = np.linalg.norm(a_cmd)
            if n > self.a_max:                                  # ivme doygunluğu
                a_cmd = a_cmd * (self.a_max / n)
            # ivme -> kısa ufuklu konum waypoint'i (kararlı PID arayüzü için)
            look = self.lookahead
            target = drone_pos + drone_vel * look + 0.5 * a_cmd * look**2
            target[2] = max(target[2], contact_z)               # pedin altını hedefleme
            return target, "YAKLASMA"

        # --- TERMINAL: kararlı takip + yumuşak alçalma ---
        if self._cmd_z is None:
            self._cmd_z = drone_pos[2]
        self._cmd_z = max(self._cmd_z - self.descent_step, contact_z)
        #target = np.array([platform_pos[0], platform_pos[1], self._cmd_z], dtype=float) Sadece konum
        lead = self.lookahead   # yaklaşık 0.4 s
        target_xy = platform_pos[:2] + platform_vel[:2] * lead   # platformun öngörülen hareketine göre, artık gerieden takip etmeyecek
        target = np.array([target_xy[0], target_xy[1], self._cmd_z])
        return target, "TERMINAL"


def position_to_action(target):
    """PID (waypoint) aksiyonu: hedef KONUM, [-1, 1] aralığına kırpılır."""
    return np.clip(target, -1.0, 1.0).astype(np.float32)
