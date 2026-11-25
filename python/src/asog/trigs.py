"""Reader for TRiGS runtime files: `.trigsbin`, manifest format `trigs.runtime` v1.

Open4DGS's `make-trigs-timeline.py` writes them: a 64-byte header (magic `TRIGSBIN`), a JSON manifest and
raw little-endian tensors. Every clip holds a `static` and a `dynamic` group of splats; a dynamic splat
carries the 3DGS attributes at its centre time plus a TRiGS motion (temporal Gaussian `times` /
`log_durations`, anchor, and the se(3) twist Bezier `zeta_*`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .container import read_manifest

MAGIC = b"TRIGSBIN"
ZETA = ["zeta_base_r", "zeta_base_t", "zeta_c0_r", "zeta_c0_t", "zeta_c1_r", "zeta_c1_t", "zeta_c2_r", "zeta_c2_t"]


@dataclass
class TrigsRuntime:
    """All splats of all clips, clip by clip, static group before dynamic group.

    Quaternions are unit wxyz, opacity is after the sigmoid, scales are log. `omega[k]` / `nu[k]` are the
    control points of the quadratic Bezier rotation / translation twist with `zeta_base_*` folded in
    (`O_k = zeta_base + zeta_ck`; the Bernstein weights sum to 1). Static splats have zero motion fields.
    """

    manifest: dict
    means: np.ndarray        # [n, 3] float64
    log_scales: np.ndarray   # [n, 3]
    quats: np.ndarray        # [n, 4] wxyz
    opacity: np.ndarray      # [n]
    sh0: np.ndarray          # [n, 3]
    shN: np.ndarray | None   # [n, coeffs, 3] float32
    times: np.ndarray        # [n] clip-local centre time (s)
    log_durations: np.ndarray  # [n] log sigma (s)
    anchors: np.ndarray      # [n, 3]
    omega: np.ndarray        # [3, n, 3]
    nu: np.ndarray           # [3, n, 3]
    static: np.ndarray       # [n] bool
    clip: np.ndarray         # [n] clip index

    @property
    def count(self) -> int:
        return len(self.means)

    def take(self, index: np.ndarray) -> TrigsRuntime:
        """The splats at `index` (a subset and/or reordering)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "manifest" or v is None:
                out[f.name] = v
            elif f.name in ("omega", "nu"):
                out[f.name] = v[:, index]
            else:
                out[f.name] = v[index]
        return TrigsRuntime(**out)


def read(path: str | Path) -> TrigsRuntime:
    """Read a `.trigsbin` file.

    Raises ValueError for a non-empty top-level static group (`--static-storage global`, the exporter's
    default): those splats belong to every clip, which `.asog` v1 cannot express (one clip per splat).
    Export with `--static-storage per-clip` instead. The checkpoints of the flip-book trainer have no
    static splats, so its exports are unaffected.
    """
    with open(path, "rb") as f:
        man, header = read_manifest(f, MAGIC)
        if man.get("format") != "trigs.runtime" or man.get("version") != 1:
            raise ValueError(f"{path}: manifest format {man.get('format')!r} v{man.get('version')}, expected trigs.runtime v1")
        if (man.get("static") or {}).get("count", 0) > 0:
            raise ValueError(f"{path}: has a global static group, which .asog v1 cannot hold; "
                             "export with make-trigs-timeline.py --static-storage per-clip")
        xyzw = man.get("semantics", {}).get("quaternionOrder") == "xyzw"

        def tensor(t: dict) -> np.ndarray:
            f.seek(header.payload_offset + t["byteOffset"])
            return np.frombuffer(f.read(t["byteLength"]), dtype=np.dtype(t["dtype"])).reshape(t["shape"])

        groups = [(ci, kind, clip[kind]) for ci, clip in enumerate(man["clips"]) for kind in ("static", "dynamic")
                  if (clip.get(kind) or {}).get("count", 0) > 0]
        n = sum(g["count"] for _, _, g in groups)
        coeffs = max([g["tensors"]["shN"]["shape"][1] for _, _, g in groups if "shN" in g["tensors"]] or [0])

        means, log_scales, sh0, anchors = (np.zeros((n, 3)) for _ in range(4))
        quats = np.zeros((n, 4))
        opacity, times, log_durations = np.zeros(n), np.zeros(n), np.zeros(n)
        shN = np.zeros((n, coeffs, 3), np.float32) if coeffs else None
        omega, nu = np.zeros((3, n, 3)), np.zeros((3, n, 3))
        static, clip_index = np.zeros(n, bool), np.zeros(n, np.int64)
        b = 0
        for ci, kind, g in groups:
            T, c = g["tensors"], g["count"]
            s = slice(b, b + c)
            means[s] = tensor(T["means"]).reshape(c, 3)
            log_scales[s] = tensor(T["scales"]).reshape(c, 3)
            q = tensor(T["quats"]).reshape(c, 4).astype(np.float64)
            quats[s] = q[:, [3, 0, 1, 2]] if xyzw else q
            opacity[s] = 1 / (1 + np.exp(-tensor(T["opacities"]).reshape(c).astype(np.float64)))
            sh0[s] = tensor(T["sh0"]).reshape(c, 3)
            if shN is not None and "shN" in T:
                k = T["shN"]["shape"][1]
                shN[s, :k] = tensor(T["shN"]).reshape(c, k, 3)
            clip_index[s] = ci
            if kind == "static":
                static[s] = True
            else:
                times[s] = tensor(T["times"]).reshape(c)
                log_durations[s] = tensor(T["log_durations"]).reshape(c)
                anchors[s] = tensor(T["anchors"]).reshape(c, 3)
                z = {k: tensor(T[k]).reshape(c, 3).astype(np.float64) for k in ZETA}
                for k in range(3):
                    omega[k, s] = z["zeta_base_r"] + z[f"zeta_c{k}_r"]
                    nu[k, s] = z["zeta_base_t"] + z[f"zeta_c{k}_t"]
            b += c
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return TrigsRuntime(man, means, log_scales, quats, opacity, sh0, shN, times, log_durations, anchors,
                        omega, nu, static, clip_index)


def temporal_threshold(man: dict) -> float | None:
    """The temporal opacity threshold the source was trained and exported with, when its temporal weight is
    the renormalised `max((gamma - th) / (1 - th), 0)` (`temporalOpacityMode: smooth_cull_renormalized`)."""
    sem = man.get("semantics", {})
    if sem.get("temporalOpacityMode") == "smooth_cull_renormalized" and "temporalCullThreshold" in sem:
        return float(sem["temporalCullThreshold"])
    return None
