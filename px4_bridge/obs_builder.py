"""PX4 biçimindeki durumdan politikanın 63'lü gözlemini kurar (LandingAviary._computeObs ile BİREBİR aynı sıra)."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from px4_bridge.frames import enu_from_ned, R_from_q_wxyz, R_enu_flu_from_R_ned_frd, T_FLU_FRD

N_ACT, ACT_DIM = 15, 3          # BaseRLAviary: ACTION_BUFFER_SIZE = ctrl_freq // 2 = 15


class ObsBuilder:
    def __init__(self, origin_enu=(0.0, 0.0, 0.0)):
        """origin_enu: Gazebo'da PyBullet dünya orijinine karşılık gelen nokta (ENU). Tekrar oynatmada 0."""
        self.origin = np.asarray(origin_enu, dtype=np.float64)
        self.reset()

    def reset(self):
        self.buf = deque([np.zeros(ACT_DIM)] * N_ACT, maxlen=N_ACT)   # env reset'teki sıfır doldurma

    def push_action(self, a):
        """Aksiyonu uyguladığın adımda, gözlemi kurmadan ÖNCE çağır (env de böyle yapıyor)."""
        self.buf.append(np.asarray(a, dtype=np.float64).reshape(ACT_DIM))

    def build(self, p_ned, v_ned, q_ned_frd_wxyz, w_frd, rel_pos_enu, rel_vel_enu):
        R = R_enu_flu_from_R_ned_frd(R_from_q_wxyz(q_ned_frd_wxyz))
        pos = enu_from_ned(p_ned) - self.origin
        rpy = Rotation.from_matrix(R).as_euler("xyz")      # PyBullet getEulerFromQuaternion ile aynı sıra
        vel = enu_from_ned(v_ned)
        w_world = R @ (T_FLU_FRD @ np.asarray(w_frd, dtype=np.float64))   # gövde FRD → dünya ENU
        return np.concatenate([pos, rpy, vel, w_world, *self.buf,
                               np.asarray(rel_pos_enu, np.float64), np.asarray(rel_vel_enu, np.float64)]).astype(np.float32)