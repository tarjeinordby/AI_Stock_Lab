"""
Signal publication finalization records.

A signal_publication record is written AFTER a successful git push,
by the `finalize_signal` step in signal.yml. A subsequent publication_ack
record is written after the final push (Phase 4) to confirm the finalized
signal file itself is committed, recording the finalized_commit_sha.

signal_validator.py checks both records: signal_publication must have
workflow_conclusion=="success" and publication_ack must be present with
a non-empty finalized_commit_sha. The signal file alone is not sufficient
— anyone can set publication_status="published" locally.

Records are append-only. Two record types per signal_run_id:
  signal_publication  — written after Phase 2 push (draft SHA)
  publication_ack     — written after Phase 4 push (finalized SHA)
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


def _sha256_json(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def compute_publication_content_hash(record: dict) -> str:
    """
    SHA-256 of a signal_publication record.
    Excludes content_hash AND finalized_commit_sha so that merging the ack's
    finalized_commit_sha into the merged dict does not invalidate this hash.
    """
    r = {k: v for k, v in record.items() if k not in ("content_hash", "finalized_commit_sha")}
    return _sha256_json(r)


def compute_ack_content_hash(record: dict) -> str:
    """
    SHA-256 of a publication_ack record.
    Excludes only content_hash — finalized_commit_sha IS covered (it's the key protected field).
    """
    r = {k: v for k, v in record.items() if k != "content_hash"}
    return _sha256_json(r)


def _append_record(record: dict) -> None:
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


def write_signal_publication(
    signal_run_id: str,
    signal_path: str,
    commit_sha: str,
    published_at: str,
    workflow_conclusion: str = "success",
) -> None:
    """
    Append a signal_publication record after the draft push (Phase 2).
    Idempotent: if a signal_publication already exists for this signal_run_id, no-op.

    Raises PublicationWriteError on I/O failure.
    """
    existing = get_signal_publication(signal_run_id)
    if existing is not None and existing.get("record_type") == "signal_publication":
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
    record["content_hash"] = compute_publication_content_hash(record)
    _append_record(record)


def write_publication_ack(signal_run_id: str, finalized_commit_sha: str) -> None:
    """
    Append a publication_ack record after the final push (Phase 4).
    Records the SHA of the commit that contains the finalized signal file.
    Idempotent: if a publication_ack already exists for this signal_run_id, no-op.

    Raises PublicationWriteError on I/O failure.
    """
    if not finalized_commit_sha:
        raise PublicationWriteError("finalized_commit_sha er tom — ack kan ikke skrives")

    if not PUBLICATIONS_FILE.exists():
        raise PublicationWriteError(
            f"Finner ikke {PUBLICATIONS_FILE} — kjør finalize_signal før ack"
        )

    # Idempotency: check for existing ack
    try:
        with open(PUBLICATIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if (
                        rec.get("signal_run_id") == signal_run_id
                        and rec.get("record_type") == "publication_ack"
                    ):
                        return  # already acked
                except json.JSONDecodeError:
                    pass
    except OSError as exc:
        raise PublicationWriteError(f"Kunne ikke lese publikasjonsfil: {exc}") from exc

    record = {
        "schema_version": 1,
        "record_type": "publication_ack",
        "signal_run_id": signal_run_id,
        "finalized_commit_sha": finalized_commit_sha,
        "written_at": _utc_now(),
        "content_hash": "",
    }
    record["content_hash"] = compute_ack_content_hash(record)
    _append_record(record)


def get_signal_publication(signal_run_id: str) -> dict | None:
    """
    Return the merged publication record for signal_run_id, or None if not found.

    Scans both signal_publication and publication_ack records for the given
    signal_run_id and returns a merged dict with finalized_commit_sha included
    if the ack record exists.
    """
    if not PUBLICATIONS_FILE.exists():
        return None
    initial = None
    ack = None
    try:
        with open(PUBLICATIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("signal_run_id") != signal_run_id:
                        continue
                    rtype = rec.get("record_type")
                    if rtype == "signal_publication":
                        initial = rec
                    elif rtype == "publication_ack":
                        ack = rec
                except json.JSONDecodeError:
                    pass
    except OSError:
        return None

    if initial is None:
        return None

    result = dict(initial)
    if ack:
        ack_stored_hash = ack.get("content_hash", "")
        if not ack_stored_hash:
            raise PublicationWriteError(
                f"publication_ack mangler content_hash for {signal_run_id} — "
                "integritet kan ikke verifiseres"
            )
        # Use compute_ack_content_hash — protects finalized_commit_sha
        ack_expected = compute_ack_content_hash(ack)
        if ack_expected != ack_stored_hash:
            raise PublicationWriteError(
                f"publication_ack content_hash er ugyldig for "
                f"{signal_run_id} — ack-recorden kan ha blitt modifisert"
            )
        result["finalized_commit_sha"] = ack.get("finalized_commit_sha", "")
    return result
