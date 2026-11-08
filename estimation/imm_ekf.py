import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise

H = np.array([[1,0, 0,0, 0,0],    # px
              [0,0, 1,0, 0,0],    # py
              [0,0, 0,0, 1,0]])   # pz 3. ekseni de dahil ettik

def _cv_filter(dt, process_var, meas_var):
    """Mod 1: her eksen sabit hız. [px,vx,py,vy,pz,vz]"""
    kf = KalmanFilter(dim_x=6, dim_z=3)
    cv = np.array([[1, dt],[0, 1]])
    kf.F = block_diag(cv, cv, cv)
    kf.H = H
    kf.R = np.eye(3)*meas_var
    q = Q_discrete_white_noise(dim=2, dt=dt, var=process_var)
    kf.Q = block_diag(q, q, q)
    kf.P *= 10.0
    return kf

def _osc_filter(dt, omega, process_var, meas_var):
    """Mod 2: x CV; y ve z harmonik (aynı omega = aynı deniz durumu)"""
    c, s = np.cos(omega*dt), np.sin(omega*dt)
    cv  = np.array([[1, dt],[0, 1]])
    osc = np.array([[c, s/omega],[-omega*s, c]])
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = block_diag(cv, osc, osc)      # x:CV, y:harmonik, z:harmonik
    kf.H = H
    kf.R = np.eye(3)*meas_var
    q = Q_discrete_white_noise(dim=2, dt=dt, var=process_var)
    kf.Q = block_diag(q, q, q)
    kf.P *= 10.0
    return kf

class PlatformIMM:
    def __init__(self, dt, wave_hz=0.3, process_var=0.05, meas_var=0.04):
        omega = 2*np.pi*wave_hz
        cv  = _cv_filter(dt, process_var, meas_var)
        osc = _osc_filter(dt, omega, process_var, meas_var)
        mu = np.array([0.5, 0.5]); M = np.array([[0.97,0.03],[0.03,0.97]])
        self.imm = IMMEstimator([cv, osc], mu, M)

    def step(self, z):
        """z=[px,py,pz] ==> (6-durum kestirim, 2 mod)."""
        self.imm.predict(); self.imm.update(np.asarray(z, dtype=float))
        return self.imm.x.copy(), self.imm.mu.copy()

    def predict_only(self):
        """Ölçüm yokken (dropout): sadece tahmin ilerlet, güncelleme yok."""
        self.imm.predict()
        return self.imm.x.copy()