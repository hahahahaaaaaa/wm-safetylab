"""Fetch the fixed upstream Wan archive without modifying an existing checkout."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx

COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
ARCHIVE_SHA256 = "2e0e8d43e92bc697ebd26394feb9e70b4672a49f9b581a8fd1f1176fe91ae427"
REPOSITORY = "https://github.com/Wan-Video/Wan2.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
# Existing deployment layout: <workspace>/deployments/wm-safetylab.
DEFAULT_REPOS_DIR = PROJECT_ROOT.parent.parent / "repos"


def fetch_source(repos_dir: Path, output_dir: Path) -> dict:
    """Download and hash the pinned archive, then create a fresh source directory."""
    repos_dir = repos_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_name = f"Wan2.1-{COMMIT}"
    source = repos_dir / source_name
    archive = output_dir / f"{source_name}.zip"
    lock_path = output_dir / "source-lock.json"
    for path in (source, archive, lock_path):
        if path.exists():
            raise FileExistsError(f"Preserving existing path: {path}; use its existing source lock")

    with httpx.Client(timeout=45, follow_redirects=True) as client:
        response = client.get(f"https://codeload.github.com/Wan-Video/Wan2.1/zip/{COMMIT}")
        response.raise_for_status()
        content = response.content
    archive_sha256 = hashlib.sha256(content).hexdigest()
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError("Pinned Wan archive SHA256 mismatch; no archive or source was written")

    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        for name in zipped.namelist():
            target = (repos_dir / name).resolve()
            if not target.is_relative_to(source) or name.split("/", 1)[0] != source_name:
                raise ValueError(f"Archive entry is outside the pinned source directory: {name}")
        # Check again after network I/O and use exclusive creation for owned paths.
        if source.exists() or archive.exists() or lock_path.exists():
            raise FileExistsError("A target appeared during download; preserving existing files")
        output_dir.mkdir(parents=True, exist_ok=True)
        repos_dir.mkdir(parents=True, exist_ok=True)
        source.mkdir()
        with archive.open("xb") as stream:
            stream.write(content)
        zipped.extractall(repos_dir)

    lock = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "archive_sha256": archive_sha256,
        "source_dir": str(source),
        "files": {
            path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*")) if path.is_file()
        },
    }
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
        stream.write("\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS_DIR,
                        help="Upstream checkout parent (default: <workspace>/repos)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Archive and source-lock.json directory (default: native/wan)")
    args = parser.parse_args()
    lock = fetch_source(args.repos_dir, args.output_dir)
    print(json.dumps({"commit": lock["commit"], "source_dir": lock["source_dir"],
                      "files": len(lock["files"])}))


if __name__ == "__main__":
    main()
