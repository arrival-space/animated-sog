"""TRiGS motion maths (format.md section 4.1), vectorised over splats.

Shared by the decoder (`Asog.evaluate`) and the error report against the source (`asog.check`).
"""

from __future__ import annotations

import numpy as np

TEMPORAL_THRESHOLD = 0.05   # opacity cut of the temporal Gaussian (default of `motion.threshold`)
SH_C0 = 0.28209479177387814


def temporal_weight(dt: np.ndarray, sigma: np.ndarray, threshold: float = TEMPORAL_THRESHOLD) -> np.ndarray:
    """Opacity factor of a dynamic splat `dt` seconds from its centre time."""
    gamma = np.exp(-0.5 * (dt / sigma) ** 2)
    return np.maximum((gamma - threshold) / (1 - threshold), 0)


def life_tau(dt: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Bezier parameter over the splat's life `[mu - 2 sigma, mu + 2 sigma]`, clamped to [0, 1]."""
    return np.clip((dt + 2 * sigma) / (4 * sigma), 0, 1)


def bezier2(tau: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Quadratic Bezier with control points `c[0..2]` (each `[n, ...]`) at `tau` `[n]`."""
    t = tau.reshape(tau.shape + (1,) * (c[0].ndim - 1))
    u = 1 - t
    return u * u * c[0] + 2 * u * t * c[1] + t * t * c[2]


def bezier4(tau: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Quartic Bezier (Bernstein form) with control points `c[0..4]` at `tau`.

    With b0..b2 the quadratic weights this is the spec's
    `b0² G0 + 2 b0 b1 G1 + (b1² + 2 b0 b2) G2 + 2 b1 b2 G3 + b2² G4`.
    """
    t = tau.reshape(tau.shape + (1,) * (c[0].ndim - 1))
    u = 1 - t
    return u ** 4 * c[0] + 4 * u ** 3 * t * c[1] + 6 * u * u * t * t * c[2] + 4 * u * t ** 3 * c[3] + t ** 4 * c[4]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions `[n, 4]`."""
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], 1)


def screw(x: np.ndarray, anchor: np.ndarray, omega: np.ndarray, nu: np.ndarray, dt: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """Screw motion of centres `x` about `anchor` by the twist (omega, nu) over `dt` seconds.

    `nu` is the translation velocity as used, i.e. already reduced to its component along omega.
    Returns the moved centres `[n, 3]` and the relative rotation `q(w)` (wxyz, `[n, 4]`) to pre-multiply
    onto the splat's rotation.
    """
    oo = (omega * omega).sum(1, keepdims=True)
    ok = oo >= 1e-8
    a = np.where(ok, anchor - (anchor * omega).sum(1, keepdims=True) / np.where(ok, oo, 1) * omega, anchor)
    w = omega * dt[:, None]
    v = nu * dt[:, None]
    th = np.linalg.norm(w, axis=1, keepdims=True)
    small = th < 1e-6
    ths = np.where(small, 1, th)
    sh, ch = np.sin(0.5 * ths), np.cos(0.5 * ths)
    A = np.where(small, 1, 2 * sh * ch / ths)
    B = np.where(small, 0.5, 2 * sh * sh / ths ** 2)
    C = np.where(small, 1 / 6, (ths - 2 * sh * ch) / ths ** 3)
    r = x - a
    wr = np.cross(w, r)
    wv = np.cross(w, v)
    centre = r + A * wr + B * np.cross(w, wr) + a + v + B * wv + C * np.cross(w, wv)
    q = np.concatenate([np.cos(0.5 * th), w * np.where(small, 0.5, sh / ths)], 1)
    return centre, q
