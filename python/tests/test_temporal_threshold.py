"""The temporal opacity threshold carried from the source into `motion.threshold`."""

from __future__ import annotations

import json
import struct

import numpy as np

from asog import check, decode, encode   # encode: the function
from asog.motion import temporal_weight


def _with_semantics(src_path, dst_path, extra: dict) -> None:
    data = bytearray(src_path.read_bytes())
    mo, ml, po, pl = struct.unpack("<QQQQ", data[16:48])
    man = json.loads(data[mo:mo + ml])
    man["semantics"].update(extra)
    body = json.dumps(man).encode()
    assert len(body) + mo <= po, "manifest grew past the payload offset"
    data[mo:po] = body + b" " * (po - mo - len(body))
    data[16:48] = struct.pack("<QQQQ", mo, len(body) + (po - mo - len(body)), po, pl)
    dst_path.write_bytes(bytes(data))


def test_default_threshold_leaves_the_file_unchanged(tmp_path, synthetic_source):
    out = tmp_path / "default.asog"
    encode(synthetic_source, out)
    a = decode.read(out)
    assert "threshold" not in a.manifest["motion"]
    assert a.manifest["motion"]["sm"] == 2.45


def test_threshold_round_trip(tmp_path, synthetic_source):
    src = tmp_path / "th.trigsbin"
    _with_semantics(synthetic_source, src, {"temporalCullThreshold": 0.01,
                                            "temporalOpacityMode": "smooth_cull_renormalized"})
    out = tmp_path / "th.asog"
    encode(src, out)
    a = decode.read(out)
    assert a.manifest["motion"]["threshold"] == 0.01
    assert abs(a.manifest["motion"]["sm"] - np.sqrt(-2 * np.log(0.01))) < 1e-3
    # a splat 2.8 sigma from its centre is visible at 0.01 and was culled at 0.05
    assert temporal_weight(np.array(2.8), np.array(1.0), 0.01) > 0 == temporal_weight(np.array(2.8), np.array(1.0))
    for i in np.nonzero(~a.static)[0]:   # a splat whose 2.8-sigma point lies inside its clip's active range
        t = float(a.mu[i] + 2.8 * a.sigma[i])
        fade, local = a.clip_weights(t)
        if fade[i] > 0 and abs(local[i] - t) < 1e-12:
            assert a.evaluate(t).weights[i] > 0
            break
    else:
        raise AssertionError("no splat to test the threshold on")
    # check re-reads the source the same way: errors stay at the quantisation level
    r = check.report(src, out)
    assert r["motion_p99"] < 50, r
