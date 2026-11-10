"""
guidance/descent_gate.py — "Önce hizalan, sonra in" kapısı (PX4 precland yatay yaklaşma + görüş konisi mantığı).
step(...) True → bekle (irtifa tut, yatayda pad'e git); False → alçal. Histerezisli.
LandingAviary ve Faz 8 ROS 2 düğümü aynı sınıfı kullanır.
"""
import numpy as np

DEFAULTS = dict(
    r0=0.10,          # koni tabanı (m)
    k=0.30,           # koni eğimi: tol = r0 + k·h
    vtol=None,        # yatay bağıl hız eşiği (m/s); None = yok
    stale_max=10,     # bu kadar adım ölçüm yoksa alçalma
    commit_h=0.12,    # bunun altında kapı yok (kararlı otur)
    climb_after=None, # beklerken bu kadar adım ölçüm yoksa tırman; None = kapalı
    z_search=0.6,     # tırmanma tavanı (pad üstü, m)
)


class DescentGate:
    def __init__(self, cfg=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.hold = False

    def reset(self):
        self.hold = False

    def step(self, drone_pos, drone_vel, plat_pos_est, plat_vel_est, steps_since_meas):
        c = self.cfg
        h = drone_pos[2] - plat_pos_est[2]
        if h < c["commit_h"]:                      # son santimler: iptal yok
            self.hold = False
            return False
        horiz = np.linalg.norm(plat_pos_est[0:2] - drone_pos[0:2])
        vrel  = np.linalg.norm(plat_vel_est[0:2] - drone_vel[0:2])
        tol   = c["r0"] + c["k"] * h
        vtol  = c["vtol"]
        stale = steps_since_meas > c["stale_max"]
        if self.hold:    # bekliyor → tam hizalanınca bırak
            ok = (horiz < tol) and (vtol is None or vrel < vtol) and not stale
            self.hold = not ok
        else:            # alçalıyor → belirgin bozulunca dur (1.3× histerezis)
            bad = (horiz > 1.3 * tol) or (vtol is not None and vrel > 1.3 * vtol) or stale
            self.hold = bad
        return self.hold