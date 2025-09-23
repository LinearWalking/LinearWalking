import numpy as np
import mujoco

def state_to_mujoco(x): # MuJoCo uses world linear velocity, we use body
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, x[3:7])
    mat = mat.reshape(3, 3)
    mj_x = x.copy()
    mj_x[19:22] = mat@mj_x[19:22]
    return mj_x

def state_from_mujoco(mj_x): # MuJoCo uses world linear velocity, we use body
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, mj_x[3:7])
    mat = mat.reshape(3, 3)
    x = mj_x.copy()
    x[19:22] = mat.T@x[19:22]
    return x

def state_error(x, x0):
    """
    Compute the error between two states x and x0 for a floating base system.
    
    Arguments:
    - x, x0: State vectors. Expected format:
        [position (3), quaternion (4), other state variables...]

    Returns:
    - A vector representing the error between x and x0.
    """

    pos_error = quat_to_rot(x0[3:7]).T @ (x[0:3] - x0[0:3])
    orientation_error = quat_to_axis_angle(L_mult(x0[3:7]).T @ x[3:7])
    rest_error = x[7:] - x0[7:]

    return np.concatenate((pos_error, orientation_error, rest_error))

def apply_delta_x(x_k, delta_x):
    """
    Applies a state increment delta_x to a floating base state x_k.

    Arguments:
    - x_k: Current state vector (length N).
    - delta_x: Increment to apply (length N - 1, assumes no change to quaternion norm).

    Returns:
    - x_next: Updated state vector after applying delta_x.
    """

    x_next = np.zeros_like(x_k)
    x_next[0:3] = x_k[0:3] + quat_to_rot(x_k[3:7]) @ delta_x[0:3]
    x_next[3:7] = L_mult(x_k[3:7]) @ axis_angle_to_quat(delta_x[3:6])
    x_next[7:] = x_k[7:] + delta_x[6:]

    return x_next

def quat_to_axis_angle(q, tol=1e-12):
    """
    Return the axis-angle vector corresponding to the provided quaternion.
    """
    qs = q[0]
    qv = q[1:4]
    norm_qv = np.linalg.norm(qv)

    if norm_qv >= tol:
        theta = 2 * np.arctan2(norm_qv, qs)
        return theta * qv / norm_qv
    else:
        return 2 * qv  # Ensures gradient calculations are stable when q ≈ [1, 0, 0, 0]


def axis_angle_to_quat(omega, tol=1e-12):
    """
    Return the quaternion corresponding to the provided axis-angle vector.
    """
    norm_omega = np.linalg.norm(omega)
    if norm_omega < tol:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        half_norm = norm_omega / 2
        sinc_val = np.sinc(norm_omega / np.pi / 2)
        return np.concatenate(([np.cos(half_norm)], omega * 0.5 * sinc_val))


def quat_conjugate(q):
    """
    Return the conjugate of the given quaternion (negates the vector part).
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def skew(v):
    """
    Return a matrix M such that v × x = Mx (cross product matrix).
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def L_mult(q):
    """
    Return a matrix representation of left quaternion multiplication.
    """
    qs = q[0]
    qv = q[1:4]
    return np.block([
        [np.array([[qs]]), -qv.reshape(1, 3)],
        [qv.reshape(3, 1), qs * np.eye(3) + skew(qv)]
    ])


def R_mult(q):
    """
    Return a matrix representation of right quaternion multiplication.
    """
    qs = q[0]
    qv = q[1:4]
    return np.block([
        [np.array([[qs]]), -qv.reshape(1, 3)],
        [qv.reshape(3, 1), qs * np.eye(3) - skew(qv)]
    ])


def attitude_jacobian(q):
    """
    Return the attitude Jacobian G, defined as dq/dt = 0.5 * G * omega.
    """
    qs = q[0]
    qv = q[1:4]
    return np.block([
        [-qv.reshape(1, 3)],
        [qs * np.eye(3) + skew(qv)]
    ])


def quat_to_rot(q):
    """
    Return the rotation matrix represented by quaternion q.
    """
    qv = q[1:4]
    skew_qv = skew(qv)
    return np.eye(3) + 2 * q[0] * skew_qv + 2 * np.dot(skew_qv, skew_qv)
