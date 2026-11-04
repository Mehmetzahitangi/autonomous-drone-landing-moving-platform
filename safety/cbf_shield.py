import numpy as np

class VerticalContactCBF():

    def __init__(self, look, dt, v_touch=0.25, k=1.0, alpha=2.0, h_on=0.15, vxy_touch=0.25, k_xy=1.0):
        self.look = look
        self.dt = dt
        self.v_touch = v_touch
        self.k = k
        self.alpha = alpha
        self.h_on = h_on          # kalkan yalnız h < h_on iken aktif (terminal bölgede)

        self.vxy_touch = vxy_touch   # temasta izin verilen yatay bağıl hız
        self.k_xy = k_xy             # yükseklikle gevşeme eğimi

    def filter(self, drone_pos, drone_vel, plat_pos, plat_vel, target_pos):

        h    = drone_pos[2] - plat_pos[2]            # pedin üstü yükseklik
        hdot = drone_vel[2] - plat_vel[2]            # göreli dikey hız (iniş negatif)

        if h > self.h_on:                      # yaklaşmada (yukarıda) kalkan kapalı 
            info = {"intervened": False, "a_nom": np.nan, "a_min": np.nan, "v_max": np.nan, "h": float(h), "hdot": float(hdot)}
            return np.asarray(target_pos, dtype=float), info
        
        a_nom = 2.0 * (target_pos[2] - drone_pos[2] - drone_vel[2]*self.look) / self.look**2

        v_max = self.v_touch + self.k * max(h, 0.0)          # yüksekken gevşek, temasta 0.4
        B = v_max + hdot                                  # B>=0 , zarftan hızlı inmiyor
        a_min = -self.k * hdot - self.alpha * B               # a_z bu sınırın altına inemez

        a_safe = max(a_nom, a_min)
        safe_z     = drone_pos[2] + drone_vel[2]*self.look + 0.5*a_safe*self.look**2

        #  2. kısım: yatay HIZ bariyeri (kapalı-form, yön korunur) 
        # Komutun ürettiği yatay bağıl hız, platform hızına göre sınırlandırılır. Yön korunur, büyüklük kısılır.
        v_cmd_xy = (target_pos[0:2] - drone_pos[0:2]) / self.look - plat_vel[0:2]
        s_cmd = np.linalg.norm(v_cmd_xy)
        s_max = self.vxy_touch + self.k_xy * max(h, 0.0)       # temasta 0.25, yukarıda gevşek

        interv_xy = s_cmd > s_max
        if interv_xy:
            v_cmd_xy = v_cmd_xy * (s_max / s_cmd)              # yönü koru, büyüklüğü kıs
        safe_xy = drone_pos[0:2] + (v_cmd_xy + plat_vel[0:2]) * self.look   # platform hızını geri ekle


        safe_target = np.array([safe_xy[0], safe_xy[1], safe_z])

        info = {"intervened": bool(a_safe > a_nom + 1e-9 or interv_xy),
                "a_nom": float(a_nom), "a_min": float(a_min), "v_max": float(v_max),
                "s_cmd": float(s_cmd), "s_max": float(s_max), "interv_xy": bool(interv_xy),
                "h": float(h), "hdot": float(hdot)}
        return safe_target, info

