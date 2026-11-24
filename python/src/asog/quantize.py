"""Codebooks for the static attributes: 1D k-means codebooks (scales, sh0, SH coefficients) and the
SH palette (k-means over the 45-value degree-1..3 SH vectors).

The palette runs on a CUDA GPU through PyTorch (`pip install animated-sog[palette]`). Without one it runs
in NumPy when `splats x palette entries` is at most `NUMPY_PALETTE_LIMIT` (a few seconds per iteration
at the limit) and raises otherwise.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

PALETTE_ITERATIONS = 12
NUMPY_PALETTE_LIMIT = 2 ** 31


def codebook1d(x: np.ndarray, k: int = 256, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Sorted 1D k-means codebook of the values in `x` (fit on up to 2M samples) and each value's index."""
    x = x.ravel()
    s = x[np.random.default_rng(0).choice(len(x), min(len(x), 2_000_000), replace=False)]
    cb = np.quantile(s, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        idx = np.searchsorted((cb[1:] + cb[:-1]) / 2, s)
        sums, cnt = np.bincount(idx, s, k), np.bincount(idx, None, k)
        cb = np.sort(np.where(cnt > 0, sums / np.maximum(cnt, 1), cb))
    return cb, np.searchsorted((cb[1:] + cb[:-1]) / 2, x).astype(np.int64)


def palette_size(n: int, limit: int) -> int:
    """Palette entries for `n` splats: about one per 16 splats as a power of two, at least 64, at most `limit`."""
    return min(limit, max(64, 2 ** int(np.log2(n / 16))))


def sh_palette(x: np.ndarray, k: int, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """k-means of the rows of `x` `[n, d]` (float32) into `k` centroids: returns centroids `[k, d]`, labels `[n]`.

    `device`: "cuda", "cpu" (PyTorch on the CPU, no size limit, minutes for ~1M splats) or None (CUDA when
    available, else NumPy). Seeded, but GPU reductions are not bit-reproducible, so labels can differ slightly
    between runs.
    """
    if device not in (None, "cuda", "cpu"):
        raise ValueError(f"palette device {device!r}, expected 'cuda' or 'cpu'")
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False
    if device == "cuda" or (device is None and cuda):
        return _palette_torch(x, k)
    if device == "cpu" and torch is not None:
        log.info("  SH palette on the CPU (PyTorch)")
        return _palette_torch(x, k, "cpu")
    if len(x) * k > NUMPY_PALETTE_LIMIT:
        raise RuntimeError(f"the SH palette of {len(x)} splats x {k} entries needs PyTorch with CUDA "
                           "(pip install animated-sog[palette]); without a GPU, encode fewer splats")
    log.info("  SH palette on the CPU (no CUDA PyTorch)")
    return _palette_numpy(x, k)


def _palette_torch(x: np.ndarray, k: int, dev: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    import torch
    n = len(x)
    rows = 16384 if dev == "cuda" else max(256, min(16384, 2 ** 28 // max(k, 1)))   # CPU: <= 1 GB of distances
    X = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    C = X[torch.randperm(n, device=dev, generator=gen)[:k]].clone()
    lab = torch.empty(n, dtype=torch.long, device=dev)
    for it in range(PALETTE_ITERATIONS):
        cn = (C * C).sum(1)
        for s in range(0, n, rows):
            lab[s:s + rows] = (cn[None] - 2 * X[s:s + rows] @ C.T).argmin(1)
        if it == PALETTE_ITERATIONS - 1:
            break
        S = torch.zeros_like(C).index_add_(0, lab, X)
        cnt = torch.zeros(k, device=dev).index_add_(0, lab, torch.ones(n, device=dev))
        dead = cnt == 0   # empty clusters restart at random splats
        C = torch.where(dead[:, None], X[torch.randint(0, n, (k,), device=dev, generator=gen)], S / cnt.clamp(min=1)[:, None])
    return C.cpu().numpy().astype(np.float64), lab.cpu().numpy()


def _palette_numpy(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(x, np.float32)
    n = len(X)
    C = X[rng.permutation(n)[:k]].copy()
    lab = np.empty(n, np.int64)
    for it in range(PALETTE_ITERATIONS):
        cn = (C * C).sum(1)
        for s in range(0, n, 4096):
            lab[s:s + 4096] = (cn[None] - 2 * X[s:s + 4096] @ C.T).argmin(1)
        if it == PALETTE_ITERATIONS - 1:
            break
        S = np.zeros_like(C)
        np.add.at(S, lab, X)
        cnt = np.bincount(lab, minlength=k).astype(np.float32)
        C = np.where((cnt == 0)[:, None], X[rng.integers(0, n, k)], S / np.maximum(cnt, 1)[:, None])
    return C.astype(np.float64), lab
