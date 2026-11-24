# animated-sog (Python)

Encoder, reference decoder and tools for **Animated SOG** (`.asog`): compressed 4D Gaussian splat clips.
An `.asog` holds splats with short lives and rigid-body motion (the TRiGS model) and stores their static
attributes the way SOG v2 does, so a SOG renderer plays a frame with only the motion added. The format is
specified in [`../docs/format.md`](../docs/format.md).

```
pip install animated-sog              # numpy only: encode (small files), decode, info, check
pip install animated-sog[palette]     # + torch: SH palette on a CUDA GPU
pip install animated-sog[flipbook]    # + torch, gsplat, imageio: the flip-book pipeline
```

Install torch from the [PyTorch index](https://pytorch.org/get-started/locally/) that matches your CUDA.

## Command line

```
asog encode clip.trigsbin clip.asog [--audio sound.mp3]
asog info clip.asog
asog check clip.trigsbin clip.asog
asog flipbook stage|train|flip|export|encode|evaluate|run ...
```

`asog encode` reads a TRiGS runtime file (`.trigsbin`, `trigs.runtime` v1, written by Open4DGS's
`make-trigs-timeline.py`) and writes an `.asog`:

| option | default | meaning |
|---|---|---|
| `--qp` | 2 | motion step: file units × 10⁻³ of displacement over a splat's visible life |
| `--qa` | 8 | anchor step, file units × 10⁻³ |
| `--prune` | 0.3 | fraction of the dynamic splats to drop, least important first (opacity × area × σ) |
| `--mbits` | 14 | position bits per axis (stored 16-bit, the low bits zero) |
| `--tbits` | 11 | centre time resolution 2^−tbits s |
| `--lsbits` | 8 | log σ bits |
| `--shk` | 65536 | largest SH palette (about one entry per 16 splats) |
| `--level` | 9 | zlib level |
| `--audio` | | sound file embedded after the streams; players loop it with the clip |
| `--palette-device` | CUDA if available | `cpu` runs the SH palette with PyTorch on the CPU (minutes for 1M splats; leaves a busy GPU alone) |

Measured on Charlie (900k splats): 28.9 MB from 319 MB, 45.3 dB mean over 6 views against the original
with SH; `--qp 0.5 --prune 0`: 47.8 MB, 46.6 dB. Pruning beyond 40% loses visible detail.

`asog info` prints the manifest summary and each stream's size; `asog check` re-derives the encoder's splat
selection and order from the source and reports the errors splat by splat (centre error over each
splat's visible life, position, rotation, scale, colour, opacity, view-dependent SH).

`asog flipbook` converts a flip book (one independently trained 3DGS PLY per frame) by distillation through
the external Open4DGS TRiGS trainer (AGPL-3.0, run as a subprocess, never imported): see
[`../docs/pipeline.md`](../docs/pipeline.md).

## Library

```python
import asog

stats = asog.encode("clip.trigsbin", "clip.asog", qp=2, prune=0.3)
clip = asog.read("clip.asog")            # dequantised arrays: means, quats (wxyz), log_scales, sh0, shN, ...
frame = clip.evaluate(1.25)              # positions [n, 3], rotations [n, 4] wxyz, weights [n] at t = 1.25 s
```

`frame.weights` is the opacity factor (clip fade × temporal weight); splats with weight 0 are not drawn.
Modules: `asog.container` (header, manifest, byte-plane streams), `asog.trigs` (TRiGS runtime reader),
`asog.encode`, `asog.decode`, `asog.motion` (the motion maths), `asog.check`, `asog.info`, `asog.flipbook`.

The SH palette is a k-means over the degree 1–3 SH vectors. It runs on a CUDA GPU through PyTorch; without
one it runs in NumPy for up to 2³¹ splats × palette entries (about 250k splats) and refuses larger inputs.
GPU k-means is seeded but not bit-reproducible, so two encodes can differ in the SH streams.

## Development

```
uv venv .venv
uv pip install -e ".[test,flipbook]"     # torch / gsplat with CUDA: see above
.venv/Scripts/python -m pytest
```

The tests build a small synthetic `.trigsbin` with known motion. Tests marked `data` use the Rubloff files
and Open4DGS install on the development machine and skip without them.
