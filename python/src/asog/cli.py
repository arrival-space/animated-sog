"""The `asog` command line.

    asog encode <in.trigsbin> <out.asog> [--qp 2] [--qa 8] [--prune 0.3] [--audio sound.mp3] ...
    asog info <file.asog>
    asog check <source.trigsbin> <file.asog>
    asog flipbook stage|train|flip|export|encode|evaluate|run ...
"""

from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path

ENCODE_OPTIONS = ("qp", "qa", "prune", "mbits", "tbits", "lsbits", "shk", "level")
STAGE_OPTIONS = ("start", "count", "stride", "rotation")
TRAIN_OPTIONS = ("fps", "keyframe_stride", "target_count", "steps", "latitude_steps", "image_size", "sigma_t",
                 "recipe", "head_boost", "trainer_args")


def _encode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--qp", type=float, default=2.0, help="motion step: file units x 1e-3 of displacement over a splat's life")
    p.add_argument("--qa", type=float, default=8.0, help="anchor step, file units x 1e-3")
    p.add_argument("--prune", type=float, default=0.3, help="fraction of the least important dynamic splats to drop")
    p.add_argument("--mbits", type=int, default=14, help="position bits per axis")
    p.add_argument("--tbits", type=int, default=11, help="centre time resolution 2^-tbits s")
    p.add_argument("--lsbits", type=int, default=8, help="log sigma bits")
    p.add_argument("--shk", type=int, default=65536, help="largest SH palette")
    p.add_argument("--level", type=int, default=9, help="zlib level")
    p.add_argument("--audio", type=Path, help="sound file to embed, looped with the clip")


def _source_args(p: argparse.ArgumentParser) -> None:
    """Fixes for .trigsbin sources other than the flip-book pipeline's (which flips its checkpoint itself)."""
    p.add_argument("--palette-device", choices=("cuda", "cpu"), help="SH palette k-means device (default: CUDA if available)")


def _train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fps", type=float, default=30, help="training frame rate")
    p.add_argument("--keyframe-stride", type=int, help="every n-th frame seeds the 4D init (default 9; 1 with --recipe dense)")
    p.add_argument("--target-count", type=int, default=400_000, help="splats")
    p.add_argument("--steps", type=int, default=60_000, help="training steps")
    p.add_argument("--latitude-steps", type=int, default=6, help="rows of the 12-column training orbit")
    p.add_argument("--image-size", type=int, default=512, help="training image width and height (px)")
    p.add_argument("--sigma-t", type=float, help="initial splat lifetime in seconds (default: one keyframe gap, "
                                                "keyframe stride / fps); splats are drawn within about 4 sigma-t of "
                                                "their keyframe, so a long one keeps most of them active every frame")
    p.add_argument("--recipe", choices=("classic", "dense"), default="classic",
                   help="dense: the flip-book recipe (every frame a keyframe, random thinning, short free lifetimes, no "
                        "relocation; needs the arrival-space fork of open4dgs-trigs, see docs/pipeline.md)")
    p.add_argument("--head-boost", type=float, default=1.0, help="dense: share of splats kept for the head (3 for face close-ups)")
    p.add_argument("--trainer-args", type=shlex.split, default=[], help="extra trainer options, appended as given")


def _tool_args(p: argparse.ArgumentParser, core: bool) -> None:
    p.add_argument("--trigs-python", help="TRiGS interpreter or wrapper script (default: $ASOG_TRIGS_PYTHON)")
    if core:
        p.add_argument("--open4dgs-core", help="open4dgs-core checkout (default: $ASOG_OPEN4DGS_CORE)")


def _stage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", type=int, default=0, help="first frame (index in name order)")
    p.add_argument("--count", type=int, help="frames to take (default: to the end)")
    p.add_argument("--stride", type=int, default=1, help="take every n-th frame")
    p.add_argument("--rotation", help="rx,ry,rz degrees (extrinsic XYZ) taking the frames to y up, feet at y = 0 "
                                      "(-90,0,0 for z-up sources); rotates positions, rotations and SH")


def _pick(args: argparse.Namespace, names: tuple[str, ...]) -> dict:
    return {k: getattr(args, k) for k in names}


def cmd_encode(args: argparse.Namespace) -> None:
    from .encode import encode
    encode(args.src, args.dst, audio=args.audio,
           palette_device=args.palette_device, **_pick(args, ENCODE_OPTIONS))


def cmd_info(args: argparse.Namespace) -> None:
    from .info import summary
    print(summary(args.file))


def cmd_check(args: argparse.Namespace) -> None:
    from .check import format_report, report
    print(format_report(report(args.source, args.file)))


def cmd_stage(args: argparse.Namespace) -> None:
    from .flipbook import stage
    stage(args.source, args.out, **_pick(args, STAGE_OPTIONS))


def cmd_train(args: argparse.Namespace) -> None:
    from .flipbook import train
    print(train(args.frames, args.run_dir, trigs_python=args.trigs_python, **_pick(args, TRAIN_OPTIONS)))


def cmd_flip(args: argparse.Namespace) -> None:
    from .flipbook import flip_checkpoint
    flip_checkpoint(args.src, args.dst)


def cmd_export(args: argparse.Namespace) -> None:
    from .flipbook import export
    export(args.checkpoints, args.output, trigs_python=args.trigs_python, open4dgs_core=args.open4dgs_core)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .flipbook import evaluate
    print(evaluate(args.teacher, args.clip, rotation=args.rotation, compare_png=args.compare_png))


def cmd_run(args: argparse.Namespace) -> None:
    from .flipbook import run
    run(args.source, args.work, open4dgs_core=args.open4dgs_core, trigs_python=args.trigs_python, audio=args.audio,
        **_pick(args, STAGE_OPTIONS + TRAIN_OPTIONS + ENCODE_OPTIONS))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="asog", description="Animated SOG (.asog): compressed 4D Gaussian splat clips")
    ap.add_argument("-q", "--quiet", action="store_true", help="only warnings and results")
    sub = ap.add_subparsers(dest="command", required=True)

    def encode_parser(subparsers) -> None:
        p = subparsers.add_parser("encode", help="encode a TRiGS runtime file (.trigsbin) to .asog")
        p.add_argument("src", type=Path)
        p.add_argument("dst", type=Path)
        _encode_args(p)
        _source_args(p)
        p.set_defaults(func=cmd_encode)

    encode_parser(sub)
    p = sub.add_parser("info", help="manifest summary and stream sizes")
    p.add_argument("file", type=Path)
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("check", help="error report of an .asog against its source .trigsbin")
    p.add_argument("source", type=Path)
    p.add_argument("file", type=Path)
    p.set_defaults(func=cmd_check)

    fb = sub.add_parser("flipbook", help="per-frame PLY sequence -> .asog via the external Open4DGS TRiGS trainer")
    fsub = fb.add_subparsers(dest="step", required=True)
    p = fsub.add_parser("stage", help="pick one segment's frames into a directory")
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    _stage_args(p)
    p.set_defaults(func=cmd_stage)
    p = fsub.add_parser("train", help="run the TRiGS PLY-sequence trainer (subprocess)")
    p.add_argument("frames", type=Path)
    p.add_argument("run_dir", type=Path)
    _train_args(p)
    _tool_args(p, core=False)
    p.set_defaults(func=cmd_train)
    p = fsub.add_parser("flip", help="rotate a checkpoint 180 deg about z to y down (quaternions xyzw)")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.set_defaults(func=cmd_flip)
    p = fsub.add_parser("export", help="Open4DGS make-trigs-timeline.py: flipped checkpoints -> .trigsbin, one clip each")
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    _tool_args(p, core=True)
    p.set_defaults(func=cmd_export)
    encode_parser(fsub)
    p = fsub.add_parser("evaluate", help="held-out PSNR / flicker / jitter of an .asog against its PLY frames")
    p.add_argument("teacher", type=Path, help="directory of the clip's PLY frames")
    p.add_argument("clip", type=Path)
    p.add_argument("--rotation", default="0,0,0",
                   help="rotation taking the teacher frames to the trainer's y up (0,0,0 for staged frames)")
    p.add_argument("--compare-png", type=Path, help="write a teacher | student strip")
    p.set_defaults(func=cmd_evaluate)
    p = fsub.add_parser("run", help="stage, train, flip, export, encode and evaluate one segment")
    p.add_argument("source", type=Path, help="directory of the flip book's PLY frames")
    p.add_argument("work", type=Path, help="work directory")
    _stage_args(p)
    _train_args(p)
    _tool_args(p, core=True)
    _encode_args(p)
    p.set_defaults(func=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
