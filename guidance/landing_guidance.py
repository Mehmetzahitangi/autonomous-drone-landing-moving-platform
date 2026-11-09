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
                 v_ref=0.5,           # tgo = mesafe / v_ref için nominal yaklaşma hızı (m/s)
                 t_go_max=4.0,        # tgo ust siniri (uzakta kazanclari asiri kucultme)
                 hold_step=0.15,      # HİZALAN'da hedefin dronun en fazla bu kadar ötesine konması (m)
                 **_):
        self.t_land = t_land
        self.t_go_min = t_go_min
        self.a_max = a_max
        self.lookahead = lookahead
        self.h_switch = h_switch
        self.descent_step = descent_step
        self.contact_offset = contact_offset
        self._t = 0.0
        self._desc_off = None
        self.v_ref = v_ref
        self.t_go_max = t_go_max    

        self._holding = False
        self._hold_z  = None
        self.hold_step = hold_step

    def reset_schedule(self):
        """Yeni bölüm başında zamanı ve alçalma çizelgesini sıfırla."""
        self._t = 0.0
        self._desc_off = None
        self._holding = False
        self._hold_z  = None

    def position_command(self, drone_pos, drone_vel, platform_pos, platform_vel, dt, hold=False):
        """İstenen HEDEF KONUMU (3,) ve aktif fazı döndürür.
        hold=True: hizalanana kadar irtifa tut — güdüm saati ve terminal iniş çizelgesi DONAR."""
        if hold and not self._holding:
            self._hold_z = float(drone_pos[2])     # bekleme BAŞLADIĞI andaki irtifa (sabit)
        self._holding = hold
        if not hold:
            self._t += dt                          # beklerken saat işlemez → t_go çökmez
        h = drone_pos[2] - platform_pos[2]
        contact_z = platform_pos[2] + self.contact_offset

        if h > self.h_switch:
            if hold:
                # HİZALAN: irtifayı tut, pad'in (öngörülen) üstüne yatayda git — PX4 "horizontal approach"
                lead = self.lookahead
                txy = platform_pos[:2] + platform_vel[:2] * lead
                # Hedefi dronun en fazla hold_step ötesine koy ("havuç"): uzak konum hedefi DSL PID'i
                # sert yatırıp takla attırır (ZEM/ZEV ara noktası da hep ~0.25 m altında kalıyordu).
                d = txy - drone_pos[:2]
                n = np.linalg.norm(d)
                if n > self.hold_step:
                    txy = drone_pos[:2] + d * (self.hold_step / n)
                return np.array([txy[0], txy[1], max(self._hold_z, contact_z)]), "HIZALAN"
    
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
        lead = self.lookahead   # yaklaşık 0.4 s
        z_surf = platform_pos[2] + platform_vel[2]*lead + self.contact_offset  # öngörülen ped yüzeyi

        if self._desc_off is None:
            self._desc_off = max(drone_pos[2] - z_surf, 0.0)   # başlangıç bağıl yükseklik
        if not hold:
            self._desc_off = max(self._desc_off - self.descent_step, 0.0)  # payı kademeli kapat; beklerken donar

        target_xy = platform_pos[:2] + platform_vel[:2]*lead
        target = np.array([target_xy[0], target_xy[1], z_surf + self._desc_off])
        return target, "TERMINAL"

def position_to_action(target):
    """PID (waypoint) aksiyonu: hedef KONUM, [-1, 1] aralığına kırpılır."""
    return np.clip(target, -1.0, 1.0).astype(np.float32)
