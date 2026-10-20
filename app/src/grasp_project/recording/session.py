"""Versioned raw arrays and explicit clocks; no inferred clock alignment.

Integrity hashes detect accidental corruption, not a maliciously rewritten archive.
An interrupted writer leaves an incomplete archive, rejected by default.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

SCHEMA_VERSION = "grasp_raw_recording_v1"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}\Z")
_HEX = re.compile(r"[a-f0-9]{64}\Z")


def _name(value, field: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{field} must be a portable identifier")
    return value


def _integer(value, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _json_copy(value):
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 64:
            raise ValueError("JSON nesting exceeds 64 levels")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("metadata must contain only JSON values")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("metadata must contain finite JSON numbers")
    return json.loads(json.dumps(value, allow_nan=False))


def _object(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return _json_copy(value)


def _digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_clock(value) -> dict:
    value = _object(value, "native_clock")
    for field in ("domain", "unit"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"native_clock.{field} must be explicitly declared")
    return value


def _summary(records: list[dict]) -> dict:
    observations = sum(record["record_type"] == "observation" for record in records)
    drops = [record for record in records if record["record_type"] == "drop"]
    return {
        "record_count": len(records),
        "observation_count": observations,
        "drop_record_count": len(drops),
        "known_dropped_observations": sum(record["count"] or 0 for record in drops),
        "unknown_drop_events": sum(record["count"] is None for record in drops),
        "source_sequence_gaps": sum(
            record["source_sequence_gap"] for record in records if record["record_type"] == "observation"
        ),
    }


@dataclass(frozen=True)
class Recording:
    """Fully validated archive. Observation ``arrays`` contain NumPy arrays."""

    manifest: dict
    records: tuple[dict, ...]


class RecordingWriter:
    """Write into a new directory only; close explicitly to finalize.

    ``native_clock`` requires ``domain`` and ``unit`` and can retain arbitrary
    JSON timestamp metadata. Native timestamp arrays belong in ``arrays``.
    Host arrival clocks are observations, never evidence of clock alignment.
    """

    def __init__(self, recording_dir, *, recording_id: str, config: dict, revision: dict, metadata: dict):
        self.root = Path(recording_dir)
        metadata = _object(metadata, "metadata")
        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "recording_id": _name(recording_id, "recording_id"),
            "trial_id": _name(metadata.get("trial_id", recording_id), "trial_id"),
            "created_epoch_ns": time.time_ns(),
            "status": "incomplete",
            "clock_alignment": "not_estimated",
            "physical_validation": False,
            "config": _object(config, "config"),
            "revision": _object(revision, "revision"),
            "metadata": metadata,
            "records": [],
        }
        for field in ("config", "revision", "metadata"):
            self._manifest[f"{field}_sha256"] = _digest_json(self._manifest[field])
        self._lock = threading.RLock()
        self._closed = False
        self._failed = False
        self._ids: set[str] = set()
        self._source_sequences: dict[str, int] = {}
        self._arrival_times: dict[str, int] = {}
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "arrays").mkdir()
        self._persist()

    def _persist(self) -> None:
        self._manifest["summary"] = _summary(self._manifest["records"])
        self._manifest.pop("manifest_sha256", None)
        self._manifest["manifest_sha256"] = _digest_json(self._manifest)
        temporary = self.root / ".manifest.json.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(self._manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        temporary.replace(self.root / "manifest.json")

    def _writable(self) -> None:
        if self._closed or self._failed:
            raise ValueError("recording is closed or a previous write failed")

    def record(
        self, kind: str, arrays: dict[str, np.ndarray], *, observation_id: str,
        host_arrival_monotonic_ns: int, host_arrival_epoch_ns: int,
        native_clock: dict, metadata: dict | None = None, source_sequence: int | None = None,
    ) -> dict:
        with self._lock:
            self._writable()
            kind = _name(kind, "kind")
            observation_id = _name(observation_id, "observation_id")
            if observation_id in self._ids:
                raise ValueError("duplicate observation_id")
            arrival = _integer(host_arrival_monotonic_ns, "host_arrival_monotonic_ns")
            if arrival < self._arrival_times.get(kind, 0):
                raise ValueError("host arrival monotonic time decreased within a stream")
            epoch = _integer(host_arrival_epoch_ns, "host_arrival_epoch_ns")
            if source_sequence is not None:
                _integer(source_sequence, "source_sequence")
                if source_sequence <= self._source_sequences.get(kind, -1):
                    raise ValueError("source_sequence must increase within a stream")
            if not isinstance(arrays, dict):
                raise TypeError("arrays must be a dictionary of NumPy arrays")
            snapshots = {}
            for name, array in arrays.items():
                _name(name, "array name")
                if not isinstance(array, np.ndarray) or array.dtype.kind not in "buifc":
                    raise ValueError("raw arrays must use numeric or boolean non-object NumPy dtypes")
                snapshots[name] = array.copy()
            sequence = len(self._manifest["records"])
            record = {
                "record_type": "observation", "recording_id": self._manifest["recording_id"],
                "trial_id": self._manifest["trial_id"], "sequence": sequence,
                "kind": kind, "observation_id": observation_id,
                "host_arrival_monotonic_ns": arrival, "host_arrival_epoch_ns": epoch,
                "native_clock": _native_clock(native_clock),
                "metadata": _object({} if metadata is None else metadata, "observation metadata"),
                "source_sequence": source_sequence,
                "source_sequence_gap": (
                    source_sequence - self._source_sequences[kind] - 1
                    if source_sequence is not None and kind in self._source_sequences else 0
                ),
                "arrays": {},
            }
            try:
                for array_index, name in enumerate(sorted(snapshots)):
                    array = snapshots[name]
                    relative = f"arrays/{sequence:08d}-{array_index:04d}.npy"
                    path = self.root / relative
                    with path.open("xb") as handle:
                        np.save(handle, array, allow_pickle=False)
                    record["arrays"][name] = {
                        "path": relative, "size_bytes": path.stat().st_size,
                        "sha256": _file_digest(path), "dtype": array.dtype.str,
                        "shape": list(array.shape),
                    }
                self._manifest["records"].append(record)
                self._persist()
            except Exception:
                self._failed = True
                raise
            self._ids.add(observation_id)
            self._arrival_times[kind] = arrival
            if source_sequence is not None:
                self._source_sequences[kind] = source_sequence
            return _json_copy(record)

    def record_drop(self, stream: str, count: int | None, reason: str, host_arrival_monotonic_ns: int) -> dict:
        with self._lock:
            self._writable()
            stream = _name(stream, "stream")
            if count is not None:
                _integer(count, "count", 1)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("drop reason must be nonempty")
            arrival = _integer(host_arrival_monotonic_ns, "host_arrival_monotonic_ns")
            if arrival < self._arrival_times.get(stream, 0):
                raise ValueError("host arrival monotonic time decreased within a stream")
            record = {
                "record_type": "drop", "recording_id": self._manifest["recording_id"],
                "trial_id": self._manifest["trial_id"], "sequence": len(self._manifest["records"]),
                "stream": stream, "count": count, "reason": reason,
                "host_arrival_monotonic_ns": arrival,
            }
            try:
                self._manifest["records"].append(record)
                self._persist()
            except Exception:
                self._failed = True
                raise
            self._arrival_times[stream] = arrival
            return _json_copy(record)

    def abort(self, reason: str) -> None:
        """Permanently prevent completion after a caller-observed runtime failure."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abort reason must be nonempty")
        with self._lock:
            if self._closed:
                raise ValueError("cannot abort a completed recording")
            self._failed = True
            self._manifest["status"] = "incomplete"
            self._manifest["failure_reason"] = reason
            self._persist()

    def close(self) -> Path:
        with self._lock:
            if self._closed:
                return self.root / "manifest.json"
            self._writable()
            self._manifest["status"] = "complete"
            try:
                self._persist()
            except Exception:
                self._failed = True
                self._manifest["status"] = "incomplete"
                raise
            self._closed = True
            return self.root / "manifest.json"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _array_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("array path must be portable and relative")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or len(pure.parts) != 2 or pure.parts[0] != "arrays":
        raise ValueError("array path must be inside arrays/")
    if not _FILENAME.fullmatch(pure.parts[1]):
        raise ValueError("array filename must be portable")
    path = root.joinpath(*pure.parts)
    if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe array: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("array path escapes recording directory")
    return path


def read_recording(recording_dir, *, require_complete: bool = True) -> Recording:
    """Validate every record and array before returning any replay input.

    ``require_complete=False`` allows forensic inspection of incomplete archives;
    it never bypasses integrity, schema, ordering, or payload validation.
    """
    root = Path(recording_dir)
    manifest_path = root / "manifest.json"
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("recording and manifest must not be symlinks")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        _object(manifest, "manifest")
        digest = manifest.get("manifest_sha256")
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if not isinstance(digest, str) or digest != _digest_json(unsigned):
            raise ValueError("manifest checksum mismatch")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported recording schema")
        if manifest.get("status") not in ("incomplete", "complete"):
            raise ValueError("invalid recording status")
        if require_complete and manifest["status"] != "complete":
            raise ValueError("incomplete recording cannot be replayed")
        _name(manifest["recording_id"], "recording_id")
        _name(manifest["trial_id"], "trial_id")
        _integer(manifest["created_epoch_ns"], "created_epoch_ns")
        if manifest["clock_alignment"] != "not_estimated" or manifest["physical_validation"] is not False:
            raise ValueError("raw recording cannot assert alignment or physical validation")
        for field in ("config", "revision", "metadata"):
            _object(manifest[field], field)
            if manifest[f"{field}_sha256"] != _digest_json(manifest[field]):
                raise ValueError(f"{field} snapshot checksum mismatch")
        if manifest["trial_id"] != manifest["metadata"].get("trial_id", manifest["recording_id"]):
            raise ValueError("trial_id disagrees with metadata")
        if not isinstance(manifest["records"], list):
            raise TypeError("records must be a list")
        seen_ids, seen_paths = set(), set()
        source_sequences, arrival_times = {}, {}
        loaded = []
        for sequence, record in enumerate(manifest["records"]):
            _object(record, "record")
            if type(record["sequence"]) is not int or record["sequence"] != sequence:
                raise ValueError("record sequence must be contiguous and ordered")
            if any(record[field] != manifest[field] for field in ("recording_id", "trial_id")):
                raise ValueError("record identity disagrees with manifest")
            arrival = _integer(record["host_arrival_monotonic_ns"], "host_arrival_monotonic_ns")
            record_type = record["record_type"]
            if record_type not in ("drop", "observation"):
                raise ValueError("unsupported record type")
            stream = _name(record["kind" if record_type == "observation" else "stream"], "stream")
            if arrival < arrival_times.get(stream, 0):
                raise ValueError("host arrival monotonic time decreased within a stream")
            arrival_times[stream] = arrival
            result = _json_copy(record)
            if record_type == "drop":
                if record["count"] is not None:
                    _integer(record["count"], "count", 1)
                if not isinstance(record["reason"], str) or not record["reason"].strip():
                    raise ValueError("drop reason must be nonempty")
            else:
                observation_id = _name(record["observation_id"], "observation_id")
                if observation_id in seen_ids:
                    raise ValueError("duplicate observation_id")
                seen_ids.add(observation_id)
                _integer(record["host_arrival_epoch_ns"], "host_arrival_epoch_ns")
                _native_clock(record["native_clock"])
                _object(record["metadata"], "observation metadata")
                source_sequence = record["source_sequence"]
                gap = 0
                if source_sequence is not None:
                    _integer(source_sequence, "source_sequence")
                    if source_sequence <= source_sequences.get(stream, -1):
                        raise ValueError("source_sequence must increase within a stream")
                    if stream in source_sequences:
                        gap = source_sequence - source_sequences[stream] - 1
                    source_sequences[stream] = source_sequence
                if type(record["source_sequence_gap"]) is not int or record["source_sequence_gap"] != gap:
                    raise ValueError("source sequence gap disagrees with recorded sequence")
                if not isinstance(record["arrays"], dict):
                    raise ValueError("arrays must be an object")
                result["arrays"] = {}
                for array_index, name in enumerate(sorted(record["arrays"])):
                    descriptor = record["arrays"][name]
                    _name(name, "array name")
                    _object(descriptor, "array descriptor")
                    path = _array_path(root, descriptor["path"])
                    if descriptor["path"] in seen_paths:
                        raise ValueError("duplicate array path")
                    seen_paths.add(descriptor["path"])
                    if descriptor["path"] != f"arrays/{sequence:08d}-{array_index:04d}.npy":
                        raise ValueError("array filename disagrees with record and array name")
                    size = _integer(descriptor["size_bytes"], "size_bytes", 1)
                    if path.stat().st_size != size:
                        raise ValueError(f"array size mismatch: {path.name}")
                    sha = descriptor["sha256"]
                    if not isinstance(sha, str) or not _HEX.fullmatch(sha) or _file_digest(path) != sha:
                        raise ValueError(f"array checksum mismatch: {path.name}")
                    shape = descriptor["shape"]
                    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                        raise ValueError("invalid array shape")
                    with path.open("rb") as handle:
                        array = np.load(handle, allow_pickle=False)
                        if handle.tell() != size:
                            raise ValueError("unexpected trailing data in NumPy payload")
                    if not isinstance(array, np.ndarray) or array.dtype.kind not in "buifc":
                        raise ValueError("unsupported NumPy payload dtype")
                    if array.dtype.str != descriptor["dtype"] or list(array.shape) != shape:
                        raise ValueError("array dtype or shape mismatch")
                    array.setflags(write=False)
                    result["arrays"][name] = array
            loaded.append(result)
        _object(manifest["summary"], "summary")
        for value in manifest["summary"].values():
            _integer(value, "summary count")
        if manifest["summary"] != _summary(manifest["records"]):
            raise ValueError("recording summary disagrees with records")
        return Recording(manifest=manifest, records=tuple(loaded))
    except (OSError, KeyError, TypeError, RecursionError, EOFError) as exc:
        raise ValueError(f"invalid raw recording: {exc}") from exc
