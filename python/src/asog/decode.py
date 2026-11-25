"""Reference decoder for `.asog` v1: dequantises every stream (format.md sections 3.3 and 4.2) and
evaluates the clip at a time (sections 4.1 and 5), as the arrival.space player does on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import container
from .motion import TEMPORAL_THRESHOLD, bezier2, bezier4, life_tau, quat_mul, screw, temporal_weight


@dataclass
class Frame:
    """The splats at one time: centres `[n, 3]`, rotations `[n, 4]` (unit wxyz) and the opacity factor
    `[n]` (clip fade x temporal weight; 0 = not drawn, those splats keep their canonical pose)."""

    positions: np.ndarray
    rotations: np.ndarray
    weights: np.ndarray


@dataclass
class Asog:
    """A decoded `.asog` file. Motion fields are zero for static splats."""

    manifest: dict
    means: np.ndarray        # [n, 3] canonical centres (at each splat's centre time)
    quats: np.ndarray        # [n, 4] unit wxyz
    log_scales: np.ndarray   # [n, 3]
    sh0: np.ndarray          # [n, 3] degree-0 SH coefficient
    opacity: np.ndarray      # [n] after the sigmoid
    shN: np.ndarray | None   # [n, coeffs, 3] degree 1..3 SH from the palette
    mu: np.ndarray           # [n] clip-local centre time (s)
    sigma: np.ndarray        # [n] temporal width (s)
    anchor: np.ndarray       # [n, 3]
    omega: np.ndarray        # [3, n, 3] rotation twist control points O0..O2 (rad/s)
    g: np.ndarray            # [5, n] quartic g = nu . omega, G0..G4
    static: np.ndarray       # [n] bool
    clip: np.ndarray         # [n] clip index
    audio: bytes | None

    @property
    def count(self) -> int:
        return len(self.means)

    def clip_weights(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Per splat at timeline time `t`: the clip's fade weight and the clip-local time."""
        fps = self.manifest["timeline"]["frameRate"]
        offset, fade = np.zeros(len(self.manifest["clips"])), np.zeros(len(self.manifest["clips"]))
        for c, clip in enumerate(self.manifest["clips"]):
            G, L, R = clip["global"], clip["local"], clip.get("render", {})
            offset[c] = (G["frameStart"] - L["frameStart"]) / fps
            fade_in, fade_out = R.get("fadeInFrames", 0) / fps, R.get("fadeOutFrames", 0) / fps
            if not G["activeTimeStart"] <= t < G["activeTimeEnd"]:
                fade[c] = 0
            elif fade_in > 0 and t < G["coreTimeStart"]:
                fade[c] = _smoothstep((t - (G["coreTimeStart"] - fade_in)) / fade_in)
            elif fade_out > 0 and t >= G["coreTimeEnd"]:
                fade[c] = _smoothstep((G["coreTimeEnd"] + fade_out - t) / fade_out)
            else:
                fade[c] = 1
        return fade[self.clip], t - offset[self.clip]

    def evaluate(self, t: float) -> Frame:
        """The clip at timeline time `t` (seconds)."""
        weights, local_t = self.clip_weights(t)
        dt = local_t - self.mu
        th = self.manifest["motion"].get("threshold", TEMPORAL_THRESHOLD)
        weights = np.where(self.static, weights, weights * temporal_weight(dt, self.sigma, th))
        positions, rotations = self.means.copy(), self.quats.copy()
        m = np.nonzero(~self.static & (weights > 0))[0]
        if len(m):
            tau = life_tau(dt[m], self.sigma[m])
            om = bezier2(tau, self.omega[:, m])
            oo = (om * om).sum(1, keepdims=True)
            nu = bezier4(tau, self.g[:, m]).reshape(-1, 1) / np.maximum(oo, 1e-24) * om
            positions[m], qrel = screw(self.means[m], self.anchor[m], om, nu, dt[m])
            q = quat_mul(qrel, self.quats[m])
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            rotations[m] = q * np.where(q[:, :1] < 0, -1, 1)
        return Frame(positions, rotations, weights)


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def read(path: str | Path) -> Asog:
    """Read and dequantise an `.asog` file."""
    M, S, audio = container.read(path)
    n = M["count"]
    mins, maxs = np.array(M["means"]["mins"]), np.array(M["means"]["maxs"])
    v = mins + S["means"] / 65535 * (maxs - mins)
    means = np.sign(v) * np.expm1(np.abs(v))
    rest = (S["quats"] / 255 - 0.5) * np.sqrt(2)
    big = S["quatmode"][:, 0]
    quats = np.zeros((n, 4))
    largest = np.sqrt(np.maximum(0, 1 - (rest * rest).sum(1)))
    for m in range(4):
        sel = big == m
        quats[np.ix_(sel, [k for k in range(4) if k != m])] = rest[sel]
        quats[sel, m] = largest[sel]
    shN = None
    if "shlabels" in S:
        sn = M["shN"]
        palette = np.array(sn["codebook"])[S["shpalette"]].reshape(sn["count"], sn["coeffs"], 3)
        shN = palette[S["shlabels"][:, 0]]
    flags = S["flags"][:, 0]
    static = (flags & 1) == 1

    mo = M["motion"]
    ls = M["logsigma"]
    sigma = np.exp(ls["min"] + S["logsigma"][:, 0] / (2 ** ls["bits"] - 1) * (ls["max"] - ls["min"]))
    life = mo["sm"] * sigma
    qpv = mo["qp"] / 1000
    anchor = S["anchor"] * mo["qa"] / 1000
    step_o = qpv / np.maximum(np.linalg.norm(means - anchor, axis=1), mo["r_min"])
    w_o = [1.0, mo["midO"], 1.0]
    o_life = np.stack([S["omega"][:, 3 * k:3 * k + 3] * step_o[:, None] * w_o[k] for k in range(3)])
    mref = np.maximum(np.stack([np.linalg.norm(bezier2(np.full(n, t), o_life), axis=1)
                                for t in np.linspace(0, 1, mo["mref_samples"])]).min(0), 1e-9) * qpv
    w_g = [1.0, mo["midG"], mo["midG"], mo["midG"], 1.0]
    return Asog(
        manifest=M,
        means=means,
        quats=quats,
        log_scales=np.array(M["scales"]["codebook"])[S["scales"]],
        sh0=np.array(M["sh0"]["codebook"])[S["sh0"]],
        opacity=S["opacity"][:, 0] / 255,
        shN=shN,
        mu=S["mu"][:, 0] / M["mu"]["scale"],
        sigma=sigma,
        anchor=anchor,
        omega=o_life / life[None, :, None],
        g=np.stack([S["g"][:, j] * mref * w_g[j] / life ** 2 for j in range(5)]),
        static=static,
        clip=flags >> 1,
        audio=audio,
    )
