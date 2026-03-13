import argparse
import shutil
from pathlib import Path


def iter_video_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def delete_test_folders(video_dir: Path, dry_run: bool = False, verbose: bool = True) -> int:
    deleted = 0
    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        if verbose or dry_run:
            print(f"{'DRY-RUN ' if dry_run else ''}delete {child}")
        if not dry_run:
            shutil.rmtree(child)
        deleted += 1
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete per-test frame folders inside each video folder in frame_boundaries, "
            "leaving only the concatenated frames in the video folder root."
        )
    )
    default_root = Path(__file__).resolve().parent / "frame_boundaries"
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Frame boundaries root (default: {default_root})",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Optional video folder name to target; repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-folder output.")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print(f"Not found: {root}")
        return 1

    video_dirs = iter_video_dirs(root)
    if args.video:
        targets = set(args.video)
        video_dirs = [d for d in video_dirs if d.name in targets]

    if not video_dirs:
        print("No video folders found.")
        return 0

    total_deleted = 0
    for video_dir in sorted(video_dirs):
        if not args.quiet:
            print(f"Video: {video_dir.name}")
        total_deleted += delete_test_folders(
            video_dir, dry_run=args.dry_run, verbose=not args.quiet
        )

    if not args.quiet:
        print(f"Deleted {total_deleted} folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
