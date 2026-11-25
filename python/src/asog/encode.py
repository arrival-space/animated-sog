"""Encoder: TRiGS runtime (`.trigsbin`) -> Animated SOG (`.asog` v1).

Steps: drop the least important dynamic splats (`prune`), order the splats (statics first, then by
0.5 s birth chunk, then by Morton code, which makes neighbouring values similar for delta + zlib), quantise
the static attributes the way SOG v2 stores them and the motion in the spec's 17-value form, and write the
streams. The quantisation step of the motion is a displacement of `qp` file-millimetres over a splat's
visible life, not a velocity, so long-lived slow splats and short fast ones get the same spatial accuracy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import container, quantize, trigs
from .motion import TEMPORAL_THRESHOLD, bezier2

log = logging.getLogger(__name__)

SM = 2.45                  # visible life: +-SM sigma (where the temporal weight drops below 0.05)
MID_O, MID_G = 3.0, 2.0    # coarser steps for the middle control points (measured: free in error)
MREF_SAMPLES = 9
R_MIN = 0.02               # file units: floor of the lever arm that turns an Omega step into a displacement
ORDER_CHUNK_SECONDS = 0.5
AUDIO_MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "aac": "audio/aac"}
SOURCE_GROUP_KEYS = ("static", "dynamic", "flags")   # per-clip tensor groups of the source, not copied


@dataclass
class EncodeStats:
    source_count: int
    count: int
    bytes: int
    stream_bytes: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def importance(src: trigs.TrigsRuntime) -> np.ndarray:
    """Pruning score of a dynamic splat: opacity x its two largest axes' area x temporal width (inf for statics)."""
    ls = np.sort(src.log_scales, axis=1)
    return np.where(src.static, np.inf, src.opacity * np.exp(ls[:, 1] + ls[:, 2]) * np.exp(src.log_durations))


def splat_order(src: trigs.TrigsRuntime, count: int) -> np.ndarray:
    """Source rows of the encoded splats, in file order: every static plus the `count - statics` most
    important dynamic splats; statics first, then by 0.5 s birth chunk, then by Morton code."""
    n_static = int(src.static.sum())
    dyn = np.nonzero(~src.static)[0]
    keep_dyn = count - n_static
    if not 0 <= keep_dyn <= len(dyn):
        raise ValueError(f"cannot keep {count} of {src.count} splats ({n_static} static)")
    imp = importance(src)
    kept = np.sort(np.concatenate([np.nonzero(src.static)[0], dyn[np.argsort(imp[dyn], kind="stable")[len(dyn) - keep_dyn:]]]))
    mean = src.means[kept]
    lo, hi = mean.min(0), mean.max(0)
    g = np.clip((mean - lo) / np.maximum(hi - lo, 1e-9) * 1023, 0, 1023).astype(np.uint64)
    code = np.zeros(len(kept), np.uint64)
    for bit in range(10):
        for a in range(3):
            code |= ((g[:, a] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + a)
    chunk = np.where(src.static[kept], -1, np.floor(src.times[kept] / ORDER_CHUNK_SECONDS)).astype(np.int64)
    return kept[np.lexsort((code, chunk))]


def encode(src: str | Path, dst: str | Path, *, qp: float = 2.0, qa: float = 8.0, prune: float = 0.3,
           mbits: int = 14, tbits: int = 11, lsbits: int = 8, shk: int = 65536, level: int = 9,
           audio: str | Path | None = None) -> EncodeStats:
    """Encode the `.trigsbin` file `src` to the `.asog` file `dst`.

    qp:     motion step, file units x 1e-3 of displacement over a splat's visible life
    qa:     anchor step, file units x 1e-3
    prune:  fraction of the dynamic splats to drop, least important first (opacity x area x sigma)
    mbits:  position bits per axis (stored 16-bit with the low bits zero, which compresses)
    tbits:  centre time resolution 2^-tbits s
    lsbits: log sigma bits
    shk:    largest SH palette (the palette has about one entry per 16 splats)
    level:  zlib level
    audio:  sound file to embed (mp3, ogg, ...), looped with the clip by players

    The temporal opacity threshold is taken from the source (`semantics.temporalCullThreshold` with
    `temporalOpacityMode: smooth_cull_renormalized`) and written as `motion.threshold` (left out when it is the default 0.05).
    """
    t0 = time.time()
    source = trigs.read(src)
    threshold = trigs.temporal_threshold(source.manifest)
    threshold = TEMPORAL_THRESHOLD if threshold is None else threshold
    # visible life +-sm sigma: where the temporal weight reaches 0 (2.45 kept for the default 0.05)
    sm = SM if threshold == TEMPORAL_THRESHOLD else round(float(np.sqrt(-2 * np.log(threshold))), 4)
    n_dyn = int((~source.static).sum())
    count = source.count - int(round(prune * n_dyn))
    s = source.take(splat_order(source, count))
    log.info(f"{source.count} splats, kept {count}, SH coeffs {0 if s.shN is None else s.shN.shape[1]} ({time.time() - t0:.0f} s)")
    n = count
    streams: list[container.Stream] = []

    def add(name: str, values: np.ndarray, delta: bool | None = None) -> None:
        st = container.encode_stream(name, values, delta=delta, level=level)
        streams.append(st)
        log.info(f"  {name:10s} {st.desc['channels']} ch, {st.desc['planes']} planes, delta {int(st.desc['delta'])}: "
                 f"{len(st.data) / n:6.2f} B/splat")

    meta: dict = {}
    # ---- position: SOG log mapping, 16-bit per axis, the low 16 - mbits bits zero
    lm = np.sign(s.means) * np.log1p(np.abs(s.means))
    mmin, mmax = lm.min(0), lm.max(0)
    step = 2 ** (16 - mbits)
    qmean = np.minimum(np.round((lm - mmin) / np.maximum(mmax - mmin, 1e-12) * 65535 / step) * step, 65535).astype(np.int64)
    meta["means"] = dict(mins=mmin.tolist(), maxs=mmax.tolist())
    add("means", qmean)
    v = mmin + qmean / 65535 * (mmax - mmin)
    mean = np.sign(v) * np.expm1(np.abs(v))   # as decoded: the motion steps depend on it
    # ---- rotation: smallest three, 8-bit, index of the largest (wxyz)
    ar = np.arange(n)
    big = np.abs(s.quats).argmax(1)
    q = s.quats * np.sign(s.quats[ar, big])[:, None]
    mask = np.ones_like(q, bool)
    mask[ar, big] = False
    add("quats", np.clip(np.round((q[mask].reshape(n, 3) / np.sqrt(2) + 0.5) * 255), 0, 255))
    add("quatmode", big)
    # ---- scales / sh0: 256-entry codebooks; opacity 8-bit
    cb, idx = quantize.codebook1d(s.log_scales)
    meta["scales"] = dict(codebook=cb.tolist())
    add("scales", idx.reshape(n, 3))
    cb, idx = quantize.codebook1d(s.sh0)
    meta["sh0"] = dict(codebook=cb.tolist())
    add("sh0", idx.reshape(n, 3))
    add("opacity", np.clip(np.round(s.opacity * 255), 0, 255))
    # ---- SH degree 1-3: k-means palette, its entries through a 256-entry codebook
    if s.shN is not None:
        coeffs = s.shN.shape[1]
        k = quantize.palette_size(n, shk)
        centroids, labels = quantize.sh_palette(s.shN.reshape(n, -1), k)
        cb, cidx = quantize.codebook1d(centroids)
        meta["shN"] = dict(codebook=cb.tolist(), count=k, coeffs=coeffs)
        add("shlabels", labels)
        add("shpalette", cidx.reshape(k, coeffs * 3), delta=False)   # per entry: coefficient-major, then rgb
        log.info(f"  SH palette K={k} ({time.time() - t0:.0f} s)")
    # ---- timing
    dyn = ~s.static
    qmu = np.round(np.where(dyn, s.times, 0) * 2 ** tbits).astype(np.int64)
    add("mu", qmu)
    lsmin, lsmax = s.log_durations[dyn].min(), s.log_durations[dyn].max()
    qls = np.where(dyn, np.round((s.log_durations - lsmin) / (lsmax - lsmin) * (2 ** lsbits - 1)), 0).astype(np.int64)
    add("logsigma", qls)
    meta["mu"] = dict(scale=2 ** tbits)
    meta["logsigma"] = dict(min=float(lsmin), max=float(lsmax), bits=lsbits)
    life = sm * np.exp(lsmin + qls / (2 ** lsbits - 1) * (lsmax - lsmin))   # decoded sigma's visible life
    # ---- motion, life-scaled: the Omega control points times the life are rotation angles over the life
    qpv = qp / 1000
    qan = np.where(dyn[:, None], np.round(s.anchors * 1000 / qa), 0).astype(np.int64)
    add("anchor", qan)
    step_o = qpv / np.maximum(np.linalg.norm(mean - qan * qa / 1000, axis=1), R_MIN)
    w_o = [1.0, MID_O, 1.0]
    q_o = [np.where(dyn[:, None], np.round(s.omega[k] * life[:, None] / (step_o[:, None] * w_o[k])), 0).astype(np.int64)
           for k in range(3)]
    o_life = np.stack([q_o[k] * step_o[:, None] * w_o[k] for k in range(3)])   # as decoded
    n_life = s.nu * life[None, :, None]
    dot = lambda a, b: (a * b).sum(1)
    # g = nu . omega as a quartic in tau, from the product of the two quadratic Beziers
    g = [dot(n_life[0], o_life[0]),
         (dot(n_life[0], o_life[1]) + dot(n_life[1], o_life[0])) / 2,
         (dot(n_life[0], o_life[2]) + 4 * dot(n_life[1], o_life[1]) + dot(n_life[2], o_life[0])) / 6,
         (dot(n_life[1], o_life[2]) + dot(n_life[2], o_life[1])) / 2,
         dot(n_life[2], o_life[2])]
    # g's step: the displacement step times the smallest |omega| over the life
    mref = np.maximum(np.stack([np.linalg.norm(bezier2(np.full(n, t), o_life), axis=1)
                                for t in np.linspace(0, 1, MREF_SAMPLES)]).min(0), 1e-9)
    w_g = [1.0, MID_G, MID_G, MID_G, 1.0]
    q_g = np.stack([np.where(dyn, np.round(g[j] / (mref * qpv * w_g[j])), 0) for j in range(5)], 1).astype(np.int64)
    add("omega", np.concatenate(q_o, 1))
    add("g", q_g)
    meta["motion"] = dict(qp=qp, qa=qa, sm=sm, midO=MID_O, midG=MID_G, mref_samples=MREF_SAMPLES, r_min=R_MIN)
    if threshold != TEMPORAL_THRESHOLD:   # absent = the 0.05 default, so default files are unchanged
        meta["motion"]["threshold"] = threshold
    add("flags", s.static.astype(np.int64) | (s.clip << 1))

    man = source.manifest
    manifest = dict(format="asog", version=1, count=n, timeline=man["timeline"],
                    clips=[{k: v for k, v in c.items() if k not in SOURCE_GROUP_KEYS} for c in man["clips"]],
                    source=dict(format=man["format"], version=man["version"]), **meta)
    sound = None
    if audio is not None:
        ext = Path(audio).suffix.lstrip(".").lower()
        sound = (Path(audio).read_bytes(), AUDIO_MIME.get(ext, f"audio/{ext}"))
    total = container.write(dst, manifest, streams, sound)
    stats = EncodeStats(source.count, n, total, {st.desc["name"]: len(st.data) for st in streams}, time.time() - t0)
    log.info(f"wrote {dst}: {total / 1e6:.1f} MB, {total / n:.2f} B/splat ({stats.seconds:.0f} s)")
    return stats
