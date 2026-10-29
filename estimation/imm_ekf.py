import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise

H = np.array([[1, 0, 0, 0],    # sadece konum ölçülüyor (px, py)
              [0, 0, 1, 0]])

def _cv_filter(dt, process_var, meas_var):
    """Mod 1: sabit hız. Durum=[px,vx,py,vy]"""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, dt, 0, 0],      # px' = px + vx*dt
                     [0, 1,  0, 0],      # vx' = vx
                     [0, 0,  1, dt],     # py' = py + vy*dt
                     [0, 0,  0, 1]])
    kf.H = H
    kf.R = np.eye(2) * meas_var                     # ölçüm gürültüsü (sensör)
    q = Q_discrete_white_noise(dim=2, dt=dt, var=process_var)
    kf.Q = block_diag(q, q)                         # süreç gürültüsü (her eksen)
    kf.P *= 10.0                                     # başlangıç belirsizliği
    return kf

def _osc_filter(dt, omega, process_var, meas_var):
    """Mod 2: y ekseninde harmonik hareket (py=0 etrafında), x yine CV"""
    c, s = np.cos(omega*dt), np.sin(omega*dt)
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, dt, 0,        0],          # x: CV
                     [0, 1,  0,        0],
                     [0, 0,  c,        s/omega],    # y: basit harmonik salınım
                     [0, 0, -omega*s,  c]])         #   (SHM'nin ayrık çözümü)
    kf.H = H
    kf.R = np.eye(2) * meas_var
    q = Q_discrete_white_noise(dim=2, dt=dt, var=process_var)
    kf.Q = block_diag(q, q)
    kf.P *= 10.0
    return kf

class PlatformIMM:
    def __init__(self, dt, wave_hz=0.3, process_var=0.05, meas_var=0.04):
        omega = 2*np.pi*wave_hz
        cv  = _cv_filter(dt, process_var, meas_var)
        osc = _osc_filter(dt, omega, process_var, meas_var)
        mu = np.array([0.5, 0.5])                   # başlangıç mod olasılıkları
        M  = np.array([[0.97, 0.03],                # Markov geçiş matrisi
                       [0.03, 0.97]])
        self.imm = IMMEstimator([cv, osc], mu, M)

    def step(self, z):
        """z=[px,py] gürültülü ölçüm ==> (4-durum kestirim, 2 mod olasılığı)."""
        self.imm.predict()
        self.imm.update(np.asarray(z, dtype=float))
        return self.imm.x.copy(), self.imm.mu.copy()