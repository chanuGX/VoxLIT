import asyncio
import json
import logging
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.settings import BACKEND_DIR
from app.core.session import validate_session_id
from app.services.audio_metadata import probe_audio_duration

logger = logging.getLogger(__name__)

# Base directory for session-based custom datasets, anchored to the Backend
# directory (not the process's working directory) so the storage location is
# identical regardless of where uvicorn is launched from.
SESSIONS_BASE_DIR = BACKEND_DIR / "uploads" / "sessions"

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILES_PER_UPLOAD_REQUEST = 50
MAX_FILES_PER_DATASET = 200

_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Dataset names: non-empty, bounded, and restricted to characters that can
# never influence filesystem path resolution.
MAX_DATASET_NAME_LENGTH = 100
_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9 _-]{1,%d}$" % MAX_DATASET_NAME_LENGTH)


class InvalidDatasetName(ValueError):
    """Raised when a dataset name fails the canonical validation contract."""


class InvalidFilename(ValueError):
    """Raised when an uploaded filename fails the canonical validation contract."""


class DatasetTooLarge(ValueError):
    """Raised when a dataset already holds the maximum permitted number of files."""


class TooManyFilesInRequest(ValueError):
    """Raised when a single upload request exceeds the per-request file cap."""


class FileTooLarge(ValueError):
    """Raised when an uploaded file exceeds MAX_FILE_BYTES."""


def validate_dataset_name(dataset_name: str) -> str:
    """Reject anything that isn't a plain, path-inert display name -- no
    path separators, traversal, colons (reserved for the external
    `custom:{sid}:{name}` id format), or control characters."""

    if not dataset_name or not _DATASET_NAME_PATTERN.fullmatch(dataset_name):
        raise InvalidDatasetName(
            "Dataset name must be 1-%d characters and may only contain "
            "letters, numbers, spaces, '_' and '-'." % MAX_DATASET_NAME_LENGTH
        )
    return dataset_name


#: Characters that are either path separators on some platform or illegal
#: in a Windows filename (the project's primary development platform, per
#: CLAUDE.md, which must still behave safely on Linux too). Rejecting all
#: of them everywhere keeps stored filenames portable and avoids a raw,
#: unhandled OS-level rename error surfacing as a generic 500/soft-failure
#: instead of a clean, validated 400.
_ILLEGAL_FILENAME_CHARS = frozenset('/\\<>:"|?*')


def validate_uploaded_filename(filename: str) -> str:
    """Reject anything that isn't a plain, single-segment filename with an
    allowed audio extension. Every stored/resolved filename in this module
    is derived from a value that has passed this check -- never from a raw
    user string used as-is."""

    if not filename:
        raise InvalidFilename("Filename must not be empty.")
    if "\x00" in filename or any(ch in _ILLEGAL_FILENAME_CHARS for ch in filename):
        raise InvalidFilename(f"Filename contains illegal path characters: {filename!r}")
    if any(ord(ch) < 0x20 for ch in filename):
        raise InvalidFilename(f"Filename contains control characters: {filename!r}")
    candidate = Path(filename)
    if candidate.name != filename or candidate.name in ("", ".", ".."):
        raise InvalidFilename(f"Filename is not a valid flat filename: {filename!r}")
    extension = candidate.suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise InvalidFilename(f"Unsupported file extension: {extension!r}")
    return filename


def _is_within(base: Path, candidate: Path) -> bool:
    """True iff `candidate`'s resolved (symlink-following) path is `base`
    itself or a descendant of it. This is the containment check every
    filesystem operation below performs before touching disk, so a symlink
    planted inside a dataset directory can't be used to read/write/delete
    outside it even if the name-level validation above were ever bypassed."""

    try:
        base_resolved = base.resolve(strict=False)
        candidate_resolved = candidate.resolve(strict=False)
    except OSError:
        return False
    return candidate_resolved == base_resolved or base_resolved in candidate_resolved.parents


class _DatasetLockRegistry:
    """Process-wide registry of per-(session, dataset) asyncio.Lock objects.

    Plain dict get-or-create is safe with no extra guard because asyncio
    coroutines only yield at `await` points -- there's no `await` between
    the lookup and the insert here, so two tasks can never race to create
    two different locks for the same key.

    Sufficient only for the current single-process deployment (no
    `--workers` flag on any documented launch path in this repo). A
    multi-worker deployment would need a cross-process lock instead (e.g. a
    Redis `SET NX PX` lock, as verification's session_assets.py sweep uses)
    -- that is a known limitation, not something this registry provides.
    """

    def __init__(self) -> None:
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    def get(self, session_id: str, dataset_name: str) -> asyncio.Lock:
        key = (session_id, dataset_name)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock


_dataset_lock_registry = _DatasetLockRegistry()


class CustomDatasetManager:
    """Manages session-based custom datasets."""

    def __init__(self, session_id: str, base_dir: Optional[Path] = None):
        self.session_id = validate_session_id(session_id)
        sessions_base = base_dir if base_dir is not None else SESSIONS_BASE_DIR
        self.session_dir = sessions_base / self.session_id
        self.datasets_dir = self.session_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset_name: str) -> Path:
        validate_dataset_name(dataset_name)
        candidate = self.datasets_dir / dataset_name
        if not _is_within(self.datasets_dir, candidate):
            raise InvalidDatasetName(
                f"Dataset name resolves outside the session dataset root: {dataset_name!r}"
            )
        return candidate

    @staticmethod
    def _write_metadata_atomic(dataset_dir: Path, metadata: Dict) -> None:
        metadata_file = dataset_dir / "dataset_metadata.json"
        tmp_file = dataset_dir / f".tmp-{secrets.token_hex(8)}.json"
        try:
            with tmp_file.open("w") as f:
                json.dump(metadata, f, indent=2)
            tmp_file.replace(metadata_file)
        except Exception:
            tmp_file.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_metadata(dataset_dir: Path) -> Optional[Dict]:
        metadata_file = dataset_dir / "dataset_metadata.json"
        if not metadata_file.exists():
            return None
        try:
            with metadata_file.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read metadata for dataset {dataset_dir.name}: {e}")
            return None

    def create_dataset(self, dataset_name: str) -> Dict:
        """Create a new custom dataset"""
        dataset_dir = self._dataset_dir(dataset_name)

        if dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' already exists in this session")

        dataset_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "files": [],
            "total_files": 0,
        }
        self._write_metadata_atomic(dataset_dir, metadata)

        logger.info(f"Created custom dataset '{dataset_name}' for session {self.session_id}")
        return metadata

    @staticmethod
    def _allocate_filename(dataset_dir: Path, filename: str) -> str:
        """Compute the deduplicated stored filename against the given
        (already fresh, lock-held) directory state."""
        file_path = dataset_dir / filename
        counter = 1
        name, ext = Path(filename).stem, Path(filename).suffix
        while file_path.exists():
            candidate = f"{name}_{counter}{ext}"
            validate_uploaded_filename(candidate)
            file_path = dataset_dir / candidate
            counter += 1
        return file_path.name

    @staticmethod
    async def _write_upload_chunked(upload: UploadFile, temp_path: Path) -> int:
        """Streams `upload` to `temp_path` in fixed-size chunks, aborting the
        moment the running total exceeds MAX_FILE_BYTES so an oversized file
        is never fully buffered or fully written before being rejected."""
        total_bytes = 0
        with temp_path.open("wb") as f:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_BYTES:
                    raise FileTooLarge(
                        f"Audio file exceeds the {MAX_FILE_BYTES // (1024 * 1024)}MB limit: "
                        f"{upload.filename}"
                    )
                f.write(chunk)
        if total_bytes == 0:
            raise ValueError(f"Audio file is empty: {upload.filename}")
        return total_bytes

    async def add_file_to_dataset_stream(
        self, dataset_name: str, filename: str, upload: UploadFile
    ) -> Dict:
        """Owns the entire read -> validate -> write -> finalize sequence for
        one uploaded file, under a per-dataset lock that covers re-reading
        metadata, the capacity check, deduplicated-name allocation, the
        streamed write, and the atomic metadata update -- so two concurrent
        uploads can never share a deduplicated name or both pass a stale
        capacity check."""
        validate_uploaded_filename(filename)
        dataset_dir = self._dataset_dir(dataset_name)

        lock = _dataset_lock_registry.get(self.session_id, dataset_name)
        async with lock:
            metadata = self._read_metadata(dataset_dir)
            if metadata is None:
                raise ValueError(f"Dataset '{dataset_name}' does not exist")

            if len(metadata["files"]) >= MAX_FILES_PER_DATASET:
                raise DatasetTooLarge(
                    f"Dataset '{dataset_name}' already has the maximum of "
                    f"{MAX_FILES_PER_DATASET} files."
                )

            stored_filename = self._allocate_filename(dataset_dir, filename)
            final_path = dataset_dir / stored_filename
            if not _is_within(dataset_dir, final_path):
                raise InvalidFilename(
                    f"Filename resolves outside the dataset directory: {filename!r}"
                )

            temp_path = dataset_dir / f".tmp-{secrets.token_hex(8)}{Path(stored_filename).suffix}"
            try:
                total_bytes = await self._write_upload_chunked(upload, temp_path)
                temp_path.replace(final_path)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise

            try:
                # Probed only now that final_path is the fully-written,
                # finalized file (never the temp path mid-write).
                probe = await run_in_threadpool(probe_audio_duration, final_path)
                duration = round(probe.duration_seconds, 2) if probe else None
                sample_rate = probe.sample_rate if probe else None

                file_metadata = {
                    "filename": stored_filename,
                    "original_filename": filename,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "size": total_bytes,
                    "uploaded_at": datetime.utcnow().isoformat(),
                }
                metadata["files"].append(file_metadata)
                metadata["total_files"] = len(metadata["files"])
                self._write_metadata_atomic(dataset_dir, metadata)
            except Exception:
                # Metadata update failed after the audio file was already
                # finalized -- delete it so it never becomes an orphan that
                # dataset_metadata.json doesn't know about.
                final_path.unlink(missing_ok=True)
                raise

            logger.info(
                f"Added file '{stored_filename}' to dataset '{dataset_name}' "
                f"in session {self.session_id}"
            )
            return file_metadata

    def list_datasets(self) -> List[Dict]:
        """List all custom datasets in the session"""
        if not self.datasets_dir.exists():
            return []

        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                metadata = self._read_metadata(dataset_dir)
                if metadata is not None:
                    datasets.append(metadata)

        return datasets

    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Get metadata for a specific dataset"""
        dataset_dir = self._dataset_dir(dataset_name)
        return self._read_metadata(dataset_dir)

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a custom dataset and all its files"""
        dataset_dir = self._dataset_dir(dataset_name)

        if not dataset_dir.exists():
            return False

        # Defense in depth: prove the resolved target is exactly one dataset
        # directory under this session's dataset root -- never the root
        # itself and never anything outside it -- before removing anything.
        if dataset_dir == self.datasets_dir or not _is_within(self.datasets_dir, dataset_dir):
            raise InvalidDatasetName(
                f"Refusing to delete a path outside the session dataset root: {dataset_name!r}"
            )

        try:
            import shutil
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted custom dataset '{dataset_name}' from session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Could not delete dataset {dataset_name}: {e}")
            return False

    def resolve_file_path(self, dataset_name: str, filename: str) -> Path:
        """Resolve the full path to a file in a custom dataset"""
        validate_uploaded_filename(filename)
        dataset_dir = self._dataset_dir(dataset_name)
        file_path = dataset_dir / filename

        if not _is_within(dataset_dir, file_path) or not file_path.is_file():
            raise FileNotFoundError(f"File '{filename}' not found in dataset '{dataset_name}'")

        return file_path

    async def repair_missing_durations(self, dataset_name: str) -> Dict:
        """Recomputes duration for every file whose stored duration is
        missing, null, or non-positive, using the same probe used at upload
        time. Resolves each file through the same containment-safe
        `resolve_file_path` used everywhere else, under the same per-dataset
        lock and atomic-write mechanism as `add_file_to_dataset_stream` --
        so this never races a concurrent upload/delete and never leaves a
        partially-written metadata file. Files that still fail to probe
        (moved, deleted, genuinely corrupt) are left as `None` rather than
        being coerced into a false `0.0`."""
        dataset_dir = self._dataset_dir(dataset_name)

        lock = _dataset_lock_registry.get(self.session_id, dataset_name)
        async with lock:
            metadata = self._read_metadata(dataset_dir)
            if metadata is None:
                raise ValueError(f"Dataset '{dataset_name}' does not exist")

            changed = False
            for file_entry in metadata["files"]:
                duration = file_entry.get("duration")
                if duration is not None and duration > 0:
                    continue

                try:
                    file_path = self.resolve_file_path(dataset_name, file_entry["filename"])
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(
                        f"Could not resolve '{file_entry['filename']}' in dataset "
                        f"'{dataset_name}' for duration repair: {e}"
                    )
                    continue

                probe = await run_in_threadpool(probe_audio_duration, file_path)
                if probe is None:
                    continue

                file_entry["duration"] = round(probe.duration_seconds, 2)
                file_entry["sample_rate"] = probe.sample_rate
                changed = True

            if changed:
                self._write_metadata_atomic(dataset_dir, metadata)

            return metadata

    def get_dataset_files_as_csv_format(self, dataset_name: str) -> List[Dict[str, str]]:
        """Get dataset files in the same format as global datasets (for compatibility)"""
        metadata = self.get_dataset_metadata(dataset_name)
        if not metadata:
            return []

        csv_format_files = []
        for file_info in metadata["files"]:
            duration = file_info.get("duration")
            sample_rate = file_info.get("sample_rate")
            csv_format_files.append({
                "filename": file_info["filename"],
                # Unknown duration/sample_rate (probe failed) stays an empty
                # string here rather than the literal text "None" -- callers
                # (Transcription/Emotion's generic dataset listing) already
                # treat an empty/non-numeric string as "no value".
                "duration": "" if duration is None else str(duration),
                "sample_rate": "" if sample_rate is None else str(sample_rate),
                "size": str(file_info["size"]),
                "uploaded_at": file_info["uploaded_at"]
            })

        return csv_format_files


def get_custom_dataset_manager(session_id: str) -> CustomDatasetManager:
    """Factory function to create a CustomDatasetManager for a session"""
    return CustomDatasetManager(session_id)


def cleanup_session_datasets(session_id: str) -> bool:
    """Clean up all datasets for a session (called when session expires)"""
    validated_sid = validate_session_id(session_id)
    session_dir = SESSIONS_BASE_DIR / validated_sid

    if not session_dir.exists():
        return True

    try:
        import shutil
        shutil.rmtree(session_dir)
        logger.info(f"Cleaned up session datasets for session {validated_sid}")
        return True
    except Exception as e:
        logger.error(f"Could not cleanup session {validated_sid}: {e}")
        return False


def is_custom_dataset(dataset_name: str) -> bool:
    """Check if a dataset name indicates a custom dataset"""
    return dataset_name.startswith("custom:")


def parse_custom_dataset_name(dataset_name: str) -> tuple[str, str]:
    """Parse custom dataset name format 'custom:session_id:dataset_name'"""
    if not is_custom_dataset(dataset_name):
        raise ValueError(f"Not a custom dataset name: {dataset_name}")

    parts = dataset_name.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid custom dataset name format: {dataset_name}")

    return parts[1], parts[2]  # session_id, dataset_name


def format_custom_dataset_name(session_id: str, dataset_name: str) -> str:
    """Format a custom dataset name for external use"""
    return f"custom:{session_id}:{dataset_name}"
