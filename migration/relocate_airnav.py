#!/usr/bin/env python3
"""Relocate an AirNav checkout and rewrite embedded absolute paths.

The project contains absolute paths both in text configuration files and in
Parquet rows.  This script performs an atomic, repeatable replacement after
the assets are downloaded on another machine.
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    "__pycache__",
    "checkpoints",
    "data",
    "model_weight",
    "logs",
    "result",
    "wandb",
    "TrainPhotoData",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--old-root", default="/data1/jingyang/AirNav")
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--old-conda", default="/data1/jingyang/miniconda3")
    parser.add_argument("--new-conda", default="/nfsdata/yangjing/miniconda")
    parser.add_argument(
        "--old-hf", default="/data1/jingyang/huggingface_cache"
    )
    parser.add_argument(
        "--new-hf", default="/nfsdata/yangjing/huggingface_cache"
    )
    parser.add_argument("--old-tmp", default="/data1/jingyang/tmp")
    parser.add_argument("--new-tmp", default="/tmp/yangjing_airnav")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def replace_nested(value: Any, replacements: dict[str, str]) -> tuple[Any, bool]:
    if isinstance(value, str):
        updated = value
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        return updated, updated != value
    if isinstance(value, list):
        changed = False
        output = []
        for item in value:
            rewritten, item_changed = replace_nested(item, replacements)
            output.append(rewritten)
            changed |= item_changed
        return output, changed
    if isinstance(value, tuple):
        rewritten, changed = replace_nested(list(value), replacements)
        return tuple(rewritten), changed
    if isinstance(value, dict):
        changed = False
        output = {}
        for key, item in value.items():
            rewritten, item_changed = replace_nested(item, replacements)
            output[key] = rewritten
            changed |= item_changed
        return output, changed
    return value, False


def rewrite_text(path: Path, replacements: dict[str, str], dry_run: bool) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    updated = original
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated == original:
        return False
    if not dry_run:
        temporary = path.with_name(path.name + ".relocating")
        temporary.write_text(updated, encoding="utf-8")
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    return True


def rewrite_parquet(
    path: Path, replacements: dict[str, str], dry_run: bool
) -> bool:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows, changed = replace_nested(table.to_pylist(), replacements)
    if not changed:
        return False
    if not dry_run:
        temporary = path.with_name(path.name + ".relocating")
        rewritten = pa.Table.from_pylist(rows, schema=table.schema)
        pq.write_table(rewritten, temporary, compression="snappy")
        os.replace(temporary, path)
    return True


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")

    replacements = {
        args.old_root.rstrip("/"): args.new_root.rstrip("/"),
        args.old_conda.rstrip("/"): args.new_conda.rstrip("/"),
        args.old_hf.rstrip("/"): args.new_hf.rstrip("/"),
        args.old_tmp.rstrip("/"): args.new_tmp.rstrip("/"),
    }

    changed_text = []
    for directory, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in SKIP_PARTS]
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            if rewrite_text(path, replacements, args.dry_run):
                changed_text.append(relative)

    changed_parquet = []
    data_root = root / "data"
    if data_root.is_dir():
        for path in sorted(data_root.rglob("*.parquet")):
            if rewrite_parquet(path, replacements, args.dry_run):
                changed_parquet.append(path.relative_to(root))

    mode = "would update" if args.dry_run else "updated"
    print(f"{mode} {len(changed_text)} text files")
    for path in changed_text:
        print(f"  text: {path}")
    print(f"{mode} {len(changed_parquet)} parquet files")
    for path in changed_parquet:
        print(f"  parquet: {path}")


if __name__ == "__main__":
    main()
