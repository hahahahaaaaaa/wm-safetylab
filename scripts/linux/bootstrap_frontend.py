"""Install verified Node 22 LTS under this workspace; no system installation.

Run with the Linux platform Python. stdout is the absolute Node bin directory;
progress is sent to stderr. Node/npm are not executed by this bootstrapper.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen

ORIGIN = "https://nodejs.org/dist"
MANAGER = "wms-bootstrap-frontend-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message):
    print(message, file=sys.stderr, flush=True)


def read_url(url):
    request = Request(url, headers={"User-Agent": MANAGER})
    with urlopen(request, timeout=120) as response:
        return response.read()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_release(index):
    releases = [entry for entry in index
                if entry.get("version", "").startswith("v22.")
                and entry.get("lts") and "linux-x64" in entry.get("files", [])]
    if not releases:
        raise RuntimeError("Official index contains no Node 22 LTS linux-x64 release")
    return max(releases, key=lambda entry: tuple(map(int, entry["version"][1:].split("."))))


def archive_checksum(checksums, filename):
    matches = []
    for line in checksums.splitlines():
        pieces = line.split()
        if len(pieces) == 2 and pieces[1].lstrip("*") == filename:
            matches.append(pieces[0])
    if len(matches) != 1 or len(matches[0]) != 64:
        raise RuntimeError("Official SHASUMS256.txt has no unique matching archive checksum")
    int(matches[0], 16)
    return matches[0].lower()


def cached_download(url, destination, expected_sha256):
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise FileExistsError(f"Existing cache differs from the official archive: {destination}")
        return
    # Temporary download is explicitly inside the workspace cache directory.
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                            dir=destination.parent)
    partial = Path(temporary)
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": MANAGER})
    try:
        with os.fdopen(descriptor, "wb") as stream, urlopen(request, timeout=120) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"Downloaded Node archive checksum mismatch; retained {partial}")
        if destination.exists():
            raise FileExistsError(f"Cache appeared during download; preserving {destination}")
        partial.rename(destination)
    except BaseException:
        log(f"Download did not complete; inspect workspace temporary file: {partial}")
        raise


def validated_members(archive, root_name):
    members = archive.getmembers()
    names = set()
    symlinks = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root_name:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        if member.name in names:
            raise RuntimeError(f"Duplicate archive member: {member.name}")
        names.add(member.name)
        if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise RuntimeError(f"Unsupported archive member type: {member.name}")
        if member.issym() or member.islnk():
            if PurePosixPath(member.linkname).is_absolute():
                raise RuntimeError(f"Absolute archive link: {member.name}")
            # Symlink targets are relative to the link; hardlinks use archive-root names.
            target = posixpath.normpath(posixpath.join(str(path.parent), member.linkname)
                                       if member.issym() else member.linkname)
            target_path = PurePosixPath(target)
            if not target_path.parts or target_path.parts[0] != root_name or ".." in target_path.parts:
                raise RuntimeError(f"Archive link escapes installation: {member.name}")
            if member.issym():
                symlinks.add(path)
    # Refuse archives that write through links, including links created later in the tar.
    for member in members:
        if any(parent in symlinks for parent in PurePosixPath(member.name).parents):
            raise RuntimeError(f"Archive member traverses a symlink: {member.name}")
        if member.islnk() and PurePosixPath(member.linkname) in symlinks:
            raise RuntimeError(f"Archive hardlink targets a symlink: {member.name}")
    return members


def tree_fingerprint(directory):
    entries = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            entries.append([relative, "symlink", os.readlink(path)])
        elif path.is_file():
            entries.append([relative, "file", path.stat().st_mode & 0o111, sha256_file(path)])
        elif path.is_dir():
            entries.append([relative, "directory"])
        else:
            raise RuntimeError(f"Unexpected installed file type: {path}")
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def read_receipt(path):
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise FileExistsError(f"Preserving unexpected receipt path: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("managed_by") != MANAGER:
        raise FileExistsError(f"Preserving unrecognized installation record: {path}")
    return record


def write_receipt(path, record):
    # Only a record owned by this bootstrapper can be replaced.
    read_receipt(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def main():
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("This bootstrapper only installs Linux x86_64 Node; no download was attempted")
    if PROJECT_ROOT.parent.name != "deployments" or PROJECT_ROOT.parent.parent.name != "world-model":
        raise RuntimeError("Expected <world-model>/deployments/<project> layout; refusing to choose a system path")
    workspace = PROJECT_ROOT.parent.parent
    runtime = workspace / "runtime"
    cache = workspace / "cache" / "node"
    # Resolve before creation; reject symlinked paths that redirect files outside the workspace.
    for path in (runtime, cache):
        if not path.resolve().is_relative_to(workspace.resolve()):
            raise RuntimeError(f"Path leaves the workspace: {path}")
        path.mkdir(parents=True, exist_ok=True)
    current_receipt = runtime / "node-install.json"
    read_receipt(current_receipt)
    index_bytes = read_url(f"{ORIGIN}/index.json")
    release = latest_release(json.loads(index_bytes))
    version = release["version"]
    root_name = f"node-{version}-linux-x64"
    filename = root_name + ".tar.xz"
    url = f"{ORIGIN}/{version}/{filename}"
    checksums_bytes = read_url(f"{ORIGIN}/{version}/SHASUMS256.txt")
    checksum = archive_checksum(checksums_bytes.decode("utf-8"), filename)
    archive = cache / filename
    target = runtime / root_name
    receipt_path = runtime / (root_name + ".install.json")
    previous = read_receipt(receipt_path)
    if target.is_symlink():
        raise FileExistsError(f"Preserving existing symlink: {target}")
    if target.exists():
        if not target.is_dir() or not previous:
            raise FileExistsError(f"Preserving installation without a matching managed receipt: {target}")
        if previous.get("version") != version or previous.get("archive_sha256") != checksum:
            raise FileExistsError(f"Preserving installation with different source identity: {target}")
        if tree_fingerprint(target) != previous.get("tree_sha256"):
            raise FileExistsError(f"Preserving modified installation: {target}")
        record = previous
        log(f"Reusing verified Node {version}: {target}")
    else:
        if previous:
            raise FileExistsError(f"Receipt exists but installation is missing; inspect {receipt_path}")
        log(f"Downloading official Node {version} to {archive}")
        cached_download(url, archive, checksum)
        staging = Path(tempfile.mkdtemp(prefix=".node-stage-", dir=runtime))
        with tarfile.open(archive, "r:xz") as tar:
            members = validated_members(tar, root_name)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(staging, members=members, filter="data")
            else:
                # Older system Python: all paths, types, links and link ancestors were checked above.
                tar.extractall(staging, members=members)
        extracted = staging / root_name
        if not (extracted / "bin/node").is_file() or not (extracted / "bin/npm").exists():
            raise RuntimeError(f"Archive lacks Node/npm; inspect {staging}")
        fingerprint = tree_fingerprint(extracted)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Installation appeared during extraction; preserving {target}")
        extracted.rename(target)
        staging.rmdir()
        record = {"managed_by": MANAGER, "version": version, "lts": release["lts"],
                  "platform": "linux-x64", "archive_url": url, "archive_sha256": checksum,
                  "checksums_url": f"{ORIGIN}/{version}/SHASUMS256.txt",
                  "install_dir": str(target), "bin_dir": str(target / "bin"),
                  "tree_sha256": fingerprint}
        write_receipt(receipt_path, record)
    write_receipt(current_receipt, record)
    print(record["bin_dir"], flush=True)


if __name__ == "__main__":
    main()
