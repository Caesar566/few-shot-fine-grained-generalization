"""Import a user-obtained official NABirds archive into the project data tree."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from pattern_recognition_coursework.data.nabirds import METADATA_FILES, load_nabirds_metadata


def _safe_name(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def _members(archive: Path):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                path = _safe_name(item.filename)
                mode = item.external_attr >> 16
                if (mode & 0o170000) == 0o120000:
                    raise ValueError(f"Archive contains a symbolic link: {item.filename}")
                yield path, item.file_size, item.is_dir(), lambda item=item: source.open(item)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as source:
            for item in source:
                path = _safe_name(item.name)
                if not (item.isfile() or item.isdir()):
                    raise ValueError(f"Archive contains a link or special file: {item.name}")
                yield path, item.size, item.isdir(), lambda item=item: source.extractfile(item)
    else:
        raise ValueError("Expected an official NABirds .zip or .tar archive")


def import_archive(archive_path: Path, project_root: Path) -> dict:
    """Extract atomically after path, layout, and dataset metadata checks."""
    archive = Path(archive_path).resolve(strict=True)
    if not archive.is_file():
        raise ValueError(f"Expected a NABirds ZIP/TAR archive file, got: {archive}")
    project = Path(project_root).resolve(strict=True)
    raw = project / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    destination = raw / "nabirds"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")
    if archive.is_relative_to(destination):
        raise ValueError("Archive cannot be inside the destination directory")

    # The archive may have one wrapping directory (for example, nabirds/).
    entries = [(path, size, is_dir) for path, size, is_dir, _ in _members(archive)]
    roots = [path.parent for path, _, is_dir in entries
             if not is_dir and path.name == "classes.txt"]
    if len(roots) != 1:
        raise ValueError("Archive must contain exactly one NABirds classes.txt")
    prefix = roots[0]
    relative_files = []
    total_size = 0
    for path, size, is_dir in entries:
        if is_dir:
            continue
        if not path.is_relative_to(prefix):
            raise ValueError(f"Archive has a file outside the NABirds root: {path}")
        relative = path.relative_to(prefix)
        if not relative.parts:
            raise ValueError(f"Invalid NABirds member: {path}")
        relative_files.append(relative)
        total_size += size
    if not set(METADATA_FILES).issubset({p.as_posix() for p in relative_files}):
        raise ValueError("Archive is missing required NABirds metadata")
    if not any(p.parts[0] == "images" and len(p.parts) > 1 for p in relative_files):
        raise ValueError("Archive has no NABirds images")
    if len(set(relative_files)) != len(relative_files):
        raise ValueError("Archive has duplicate file paths")
    if total_size > shutil.disk_usage(raw).free:
        raise OSError("Insufficient free disk space for NABirds extraction")

    staging = Path(tempfile.mkdtemp(prefix=".nabirds-import-", dir=raw))
    wanted = set(relative_files)
    try:
        for path, size, is_dir, open_member in _members(archive):
            if is_dir or not path.is_relative_to(prefix):
                continue
            relative = path.relative_to(prefix)
            if relative not in wanted:
                continue
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open_member() as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.stat().st_size != size:
                raise ValueError(f"Truncated archive member: {path}")
        metadata = load_nabirds_metadata(staging)
        missing = [relative for relative in metadata["images"].values()
                   if not (staging / "images" / relative).is_file()]
        if missing:
            raise ValueError(f"Missing {len(missing)} indexed images; first: {missing[0]}")
        if destination.exists():
            raise FileExistsError(f"Destination appeared during import: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(destination), "archive_sha256": digest,
            "classes": len(metadata["classes"]), "images": len(metadata["images"]),
            "metadata_sha256": metadata["hashes"]}
