"""
Signal publication finalization records.

A signal_publication record is written AFTER a successful git push,
by the `finalize_signal` step in signal.yml. This is the authoritative
proof that:
  - The signal was committed and pushed to git.
  - The workflow concluded successfully.
  - The commit SHA is known and recorded.

The signal file itself is updated during finalization to set:
  publication_status = "published"
  published_at        = <timestamp>
  published_commit_sha = <commit SHA>
  signal_content_hash  = <SHA-256 of final payload>

signal_validator.py checks both the signal file's content hash AND the
presence of a publication record here. The signal file alone is not
sufficient — anyone can set publication_status="published" locally.

Records are append-only, one per signal_run_id.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PUBLICATIONS_FILE = Path("data_v4/ledger/signal_publications.jsonl")
_LOCK_FILE = Path("data_v4/ledger/signal_publications.lock")


class PublicationWriteError(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _content_hash(record: dict) -> str:
    r = {k: v for k, v in record.items() if k != "content_hash"}
    return hashlib.sha256(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def write_signal_publication(
    signal_run_id: str,
    signal_path: str,
    commit_sha: str,
    published_at: str,
    workflow_conclusion: str = "success",
) -> None:
    """
    Append a publication record after a successful git push.
    Idempotent: if signal_run_id is already in the file, no-op.

    Raises PublicationWriteError on I/O failure.
    """
    if get_signal_publication(signal_run_id) is not None:
        return

    record = {
        "schema_version": 1,
        "record_type": "signal_publication",
        "signal_run_id": signal_run_id,
        "signal_path": str(signal_path),
        "published_at": published_at,
        "commit_sha": commit_sha,
        "workflow_conclusion": workflow_conclusion,
        "written_at": _utc_now(),
        "content_hash": "",
    }
    record["content_hash"] = _content_hash(record)

    PUBLICATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch(exist_ok=True)

    try:
        with open(_LOCK_FILE, "rb") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(PUBLICATIONS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
                    f.flush()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError as exc:
        raise PublicationWriteError(f"Kunne ikke skrive publikasjonsrecord: {exc}") from exc


def get_signal_publication(signal_run_id: str) -> dict | None:
    """
    Return the publication record for signal_run_id, or None if not found.
    Scans the publications JSONL file (typically <50 records per month).
    """
    if not PUBLICATIONS_FILE.exists():
        return None
    try:
        with open(PUBLICATIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("signal_run_id") == signal_run_id:
                        return rec
                except json.JSONDecodeError:
                    pass
    except OSError:
        return None
    return None
