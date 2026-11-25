# Animated SOG (`.asog`) — format v1

An `.asog` file is a 4D Gaussian splat clip: a set of 3D Gaussians, each with a short life in time and a
rigid-body motion over that life, plus an optional soundtrack. The motion model is TRiGS
(*Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting*, arXiv 2604.00538); the static
attributes are stored the way SOG v2 stores them, so a renderer that reads SOG can render an `.asog` frame
with only the motion added.

All numbers are little-endian. "Splat" and "Gaussian" mean the same thing.

## 1. Container

```
offset 0   magic        4 bytes  "ASOG"
offset 4   reserved    12 bytes  zero
offset 16  manifestOffset   u64  byte offset of the manifest (64)
offset 24  manifestLength   u64  manifest size in bytes
offset 32  payloadOffset    u64  byte offset of the payload (64-aligned)
offset 40  payloadLength    u64  payload size in bytes
offset 48  reserved    16 bytes  zero
manifestOffset   UTF-8 JSON manifest (section 2)
payloadOffset    payload: the streams (section 3), then the soundtrack (section 6)
```

Every `byteOffset` in the manifest is relative to `payloadOffset`. The reference encoder starts every
stream and the soundtrack on an 8-byte boundary (zero padding); decoders must not rely on it. Streams appear in the payload in the
order a decoder needs them (section 3.2), so a file can be decoded while it downloads, one stream at a time.

## 2. Manifest

```jsonc
{
  "format": "asog", "version": 1,
  "count": 900946,                       // number of splats
  "timeline": { "frameRate": 30, "durationSeconds": 7.1, ... },
  "clips": [ { "global": {...}, "local": {...}, "render": {...} } ],   // section 5
  "source": { "format": "trigs.runtime", "version": 1 },             // informational
  "means":    { "mins": [3 floats], "maxs": [3 floats] },
  "scales":   { "codebook": [256 floats] },
  "sh0":      { "codebook": [256 floats] },
  "shN":      { "codebook": [256 floats], "count": K, "coeffs": 15 },  // absent without SH
  "mu":       { "scale": 2048 },
  "logsigma": { "min": float, "max": float, "bits": 8 },
  "motion":   { "qp": 2, "qa": 8, "sm": 2.45, "midO": 3, "midG": 2, "mref_samples": 9, "r_min": 0.02,
                "threshold": 0.05 },                                // threshold optional, default 0.05 (4.1)
  "audio":    { "mime": "audio/mpeg", "byteOffset": int, "byteLength": int },      // optional
  "streams":  [ { "name", "count", "channels", "planes", "delta", "zigzag", "byteOffset", "byteLength" } ]
}
```

## 3. Streams

### 3.1 Integer coding

Each stream holds an `count × channels` array of integers `x[i][c]`, stored as:

1. **Delta** (if `delta`): `x[i] = d[i] + x[i-1]` along the splat order, per channel (`x[-1] = 0`).
2. **Zigzag** (if `zigzag`): signed values were mapped `z = (x << 1) ^ (x >> 63)`; decode `x = (z >> 1) ^ -(z & 1)`.
3. **Byte planes**: `z` uses `planes` bytes. The raw bytes are plane-major, then channel-major, then splat:
   byte `p` of channel `c` of splat `i` is at `raw[(p * channels + c) * count + i]`.
4. The raw bytes are compressed with zlib (RFC 1950, "deflate" in `DecompressionStream`), `byteLength` bytes.
   The inflated size is exactly `count * channels * planes`.

### 3.2 Stream list (v1)

Order in the payload, as written by the reference encoder. `n` = `count`.

| name | count × channels | meaning |
|---|---|---|
| `means` | n × 3 | position, 16-bit SOG log mapping (3.3) |
| `quats` | n × 3 | rotation, smallest three, 8-bit (3.4) |
| `quatmode` | n × 1 | index (0–3, wxyz) of the largest quaternion component |
| `scales` | n × 3 | index into `scales.codebook` (log scale per axis) |
| `sh0` | n × 3 | index into `sh0.codebook` (DC colour coefficient per channel) |
| `opacity` | n × 1 | opacity × 255 (after the sigmoid) |
| `shlabels` | n × 1 | SH palette entry of the splat (16-bit) |
| `shpalette` | K × (coeffs·3) | per palette entry: index into `shN.codebook`, coefficient-major, then r g b |
| `mu` | n × 1 | centre time × `mu.scale` (seconds) |
| `logsigma` | n × 1 | log temporal width, `bits`-bit over `[min, max]` |
| `anchor` | n × 3 | rotation pivot × 1000 / `motion.qa` |
| `omega` | n × 9 | rotation control points O0, O1, O2 (xyz each), life-scaled (4.2) |
| `g` | n × 5 | translation quartic G0…G4, life-scaled (4.2) |
| `flags` | n × 1 | bit 0: static splat; bits 1+: clip index |

### 3.3 Static attributes (SOG v2)

- **Position**: `v = mins + q / 65535 · (maxs − mins)`, `x = sign(v) · (e^|v| − 1)` per axis.
- **Rotation**: `r_k = (q_k / 255 − 0.5) · √2` for the three stored components (in wxyz order with the
  `quatmode` component left out); the largest is `√(1 − Σ r_k²)`, positive.
- **Scale**: `exp(scales.codebook[q])` per axis.
- **Colour**: `sh0.codebook[q]` is the degree-0 SH coefficient (`colour = 0.5 + C0 · c`, `C0 = 0.28209479177387814`).
- **Opacity**: `q / 255`.
- **SH degree 1–3**: coefficient `j`, channel `c` of the splat is `shN.codebook[palette[label][j·3 + c]]`.

These are exactly the arrays of a SOG v2 file (`means_l/means_u`, `quats` with mode 252 + index, `scales`,
`sh0` with opacity in alpha, `sh_labels`, `sh_centroids` with entry `k`, coefficient `j` at texel
`((k mod 64)·coeffs + j, ⌊k / 64⌋)`), so a SOG renderer can take them unchanged.

## 4. Motion

### 4.1 Model (TRiGS)

A dynamic splat has a centre time `μ`, a temporal width `σ = exp(logsigma)`, an anchor `a`, and an se(3)
twist whose rotation part `ω(τ)` and translation part `ν(τ)` are quadratic Bézier curves over its life.
At clip-local time `t`, with `Δt = t − μ`:

```
weight  = max((γ − th) / (1 − th), 0),   γ = exp(−½ (Δt / σ)²)            opacity *= weight
           th = motion.threshold (default 0.05; Open4DGS runs train with 0.01)
τ       = clamp((t − (μ − 2σ)) / (4σ), 0, 1),  b0 = (1−τ)², b1 = 2(1−τ)τ, b2 = τ²
ω       = b0·O0 + b1·O1 + b2·O2
g       = b0²·G0 + 2b0b1·G1 + (b1² + 2b0b2)·G2 + 2b1b2·G3 + b2²·G4
a⊥      = a − (a·ω / |ω|²) ω            (only when |ω|² ≥ 1e-8, else a)
ν∥      = (g / max(|ω|², 1e-24)) ω
w = ω Δt,  v = ν∥ Δt,  θ = |w|
A = sin θ / θ,  B = (1 − cos θ) / θ²,  C = (θ − sin θ) / θ³     (1, ½, ⅙ as θ → 0)
centre  = r + A (w × r) + B w × (w × r) + a⊥ + v + B (w × v) + C w × (w × v),   r = x − a⊥
rotation = q(w) ⊗ rotation,   q(w) = (w = cos(θ/2), xyz = w/θ · sin(θ/2))
```

A static splat (flag bit 0) does not move and is always visible within its clip. `O0..O2` already contain
TRiGS's `ζ_base` term (`O_k = ζ_base,r + ζ_k,r`), and only the component of `ν` along `ω` affects the
motion, which `g = ν·ω` (a quartic in τ) captures exactly: 17 values (9 + 5 + anchor 3) per splat.

### 4.2 Dequantisation

```
μ        = q_mu / mu.scale
σ        = exp(logsigma.min + q / (2^bits − 1) · (logsigma.max − logsigma.min))
L        = motion.sm · σ                                  (≈ half-width of the visible life, seconds; the 0.05
                                                           cut is at ±2.4477σ, `sm` = 2.45 only scales the steps;
                                                           the reference encoder writes √(−2 ln th) for another th)
a        = q_anchor · motion.qa / 1000
step_O   = (motion.qp / 1000) / max(|x − a|, motion.r_min)   (x = decoded position)
O_k      = q_omega[k] · step_O · wO[k] / L,      wO = [1, midO, 1]
m_ref    = max(min over τ_s = s/(mref_samples−1) of |Σ b_k(τ_s) · O_k · L|, 1e-9) · motion.qp / 1000
G_j      = q_g[j] · m_ref · wG[j] / L²,          wG = [1, midG, midG, midG, 1]
```

`m_ref` is computed from the dequantised `O_k` (decoders that store `O` at reduced precision should compute
it from the stored values, as the encoder computes it from the dequantised ones). The quantisation step is
a displacement of `qp` millimetres (file units × 10⁻³) over a splat's visible life, not a velocity.

## 5. Clips and time

A file may hold several clips that share one timeline (e.g. takes or lighting segments); a splat belongs to
the clip in its flags. Per clip, from the TRiGS timeline exporter:

- `global.frameStart`, `local.frameStart`: the clip's local time is `t_local = t − (global.frameStart − local.frameStart) / frameRate`.
- `global.activeTimeStart` / `activeTimeEnd`: the clip's splats exist only in `[activeStart, activeEnd)`.
- `global.coreTimeStart` / `coreTimeEnd` with `render.fadeInFrames` / `fadeOutFrames`: outside the core range
  the clip fades over the fade length. v1 players always use smoothstep; `render.fadeCurve` (Open4DGS can
  export `linear`) is not honoured in v1.

Players loop `t` over `timeline.durationSeconds`. The reference player derives `t` from a shared clock
(`(serverTime mod duration)`) so every viewer sees the same frame.

## 6. Soundtrack

If `audio` is present, the sound file (`mime`) is the `byteLength` bytes at `byteOffset`, after the streams.
It loops with the clip: at clip time `t` the sound should be at `t` (plus the audio output latency). It
plays from 0 on each loop of the clip and is cut at the clip's duration; a shorter sound leaves silence
until the clip wraps. The reference players seek when the two drift more than 60 ms apart.

## 7. Coordinates

Positions are in the source's units (metres for the files we make) in TRiGS/COLMAP orientation: **y down**.
Players rotate the clip 180° about z to show it y up.
