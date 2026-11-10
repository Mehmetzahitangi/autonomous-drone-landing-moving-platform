"""Eksen dönüşümleri: ENU <-> NED (dünya), FLU <-> FRD (gövde). Matrisler kendi tersidir (T @ T = I)."""
import numpy as np
from scipy.spatial.transform import Rotation

T_ENU_NED = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])   # v_enu = T_ENU_NED @ v_ned  (ve tersi aynı matris)
T_FLU_FRD = np.diag([1.0, -1.0, -1.0])     # v_flu = T_FLU_FRD @ v_frd  (ve tersi aynı matris)


def enu_from_ned(v):
    return T_ENU_NED @ np.asarray(v, dtype=np.float64)


ned_from_enu = enu_from_ned


def R_from_q_wxyz(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def q_wxyz_from_R(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def R_enu_flu_from_R_ned_frd(R_ned_frd):
    """PX4 tutumu (gövde FRD → dünya NED) → PyBullet tutumu (gövde FLU → dünya ENU)."""
    return T_ENU_NED @ R_ned_frd @ T_FLU_FRD


R_ned_frd_from_R_enu_flu = R_enu_flu_from_R_ned_frd   # aynı formül (matrisler kendi tersi)