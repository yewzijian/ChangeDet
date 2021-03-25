import numpy as np


class Quaternion:
    """Basic quaternion class.
    We use the hamilton notation, with the real part coming first
    """
    def __init__(self, q: np.ndarray):
        assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)
        self.data = q

    @property
    def inverse(self):
        return self.__class__(self._quat_inv(self.data))

    def to_axis_angle(self):
        quat = self.data
        sin_theta = np.linalg.norm(quat[..., 1:], axis=-1)
        cos_theta = quat[..., 0]
        two_theta = 2.0 * np.arctan2(sin_theta, cos_theta) if cos_theta > 0.0 \
            else 2.0 * np.arctan2(-sin_theta, -cos_theta)

        axis = quat[..., 1:] / np.linalg.norm(quat[..., 1:], axis=-1, keepdims=True)
        angle_axis = axis * two_theta

        return angle_axis

    def rotate(self, pts: np.ndarray):
        """Rotates points

        Args:
            pts: (N, 3)
        """
        q = self.data

        zeros = np.zeros_like(pts[..., 0:1])
        v = np.concatenate([zeros, pts], axis=-1)
        if v.ndim == q.ndim:
            return self._quat_mul(self._quat_mul(q, v), self._quat_inv(q))[..., 1:4]
        elif v.ndim == q.ndim + 1:
            return self._quat_mul(self._quat_mul(q[..., None, :], v), self._quat_inv(q))[..., 1:4]
        else:
            raise AssertionError('pts should have the same or 1 more dimension as quaternion')

    @staticmethod
    def _quat_mul(q1, q2):
        qout = np.stack([
            q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3],
            q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2],
            q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1],
            q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]
        ], axis=-1)
        return qout

    @staticmethod
    def _quat_inv(q):
        return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)

    def __repr__(self):
        return str(self.data)
