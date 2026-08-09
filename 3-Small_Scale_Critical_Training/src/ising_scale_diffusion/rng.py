from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_u64(*parts: Any) -> int:
    """Return a process-independent 64-bit seed for a logical identity."""
    payload = json.dumps(
        parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def stable_i63(*parts: Any) -> int:
    return stable_u64(*parts) & ((1 << 63) - 1)
