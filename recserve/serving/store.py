"""User state in Redis: one hash per user with compact binary fields.

    user:{userId} -> recent (int32 item_idx, oldest first, last L positives)
                     seen   (int32 item_idx of every rated item, for filtering)
                     stats  (float32, USER_STATS order)

Binary arrays keep a user at a few hundred bytes to a few KB and decode with a
single np.frombuffer, which matters on the request path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recserve.ranking.features import USER_STATS

KEY = "user:{}"


@dataclass
class UserRecord:
    recent: np.ndarray  # int64 item_idx, oldest first
    seen: np.ndarray  # int64 item_idx
    stats: np.ndarray  # float32, len(USER_STATS)


def encode(recent: np.ndarray, seen: np.ndarray, stats: np.ndarray) -> dict[str, bytes]:
    return {
        "recent": np.asarray(recent, dtype=np.int32).tobytes(),
        "seen": np.asarray(seen, dtype=np.int32).tobytes(),
        "stats": np.asarray(stats, dtype=np.float32).tobytes(),
    }


def decode(fields: dict[bytes, bytes]) -> UserRecord | None:
    if not fields:
        return None
    stats = np.frombuffer(fields[b"stats"], dtype=np.float32)
    if len(stats) != len(USER_STATS):
        raise ValueError(f"stats has {len(stats)} values, expected {len(USER_STATS)}")
    return UserRecord(
        recent=np.frombuffer(fields[b"recent"], dtype=np.int32).astype(np.int64),
        seen=np.frombuffer(fields[b"seen"], dtype=np.int32).astype(np.int64),
        stats=stats,
    )
