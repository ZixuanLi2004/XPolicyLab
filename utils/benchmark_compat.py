"""MobileArena public names with compatibility for published MobileArena data."""

import os
from pathlib import Path


def mobile_arena_option(suffix, default=None):
    return os.environ.get(f"MOBILE_ARENA_{suffix}") or os.environ.get(f"MIVERSE_{suffix}") or default


def mobile_arena_raw_root(workspace=None):
    explicit = mobile_arena_option("RAW_ROOT") or mobile_arena_option("DATASET")
    if explicit:
        return Path(explicit).expanduser()
    workspace = Path(workspace) if workspace is not None else Path(__file__).resolve().parents[2]
    candidates = [
        workspace / "data" / "MobileArena",
        workspace / "data" / "MiVerse",
        workspace / "dataset" / "MiVerse" / "data" / "MiVerse",
    ]
    return next((path for path in candidates if path.is_dir()), candidates[0])


if __name__ == "__main__":
    print(mobile_arena_raw_root())
