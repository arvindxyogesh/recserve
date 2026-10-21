import numpy as np
import pytest

from recserve.ranking.features import USER_STATS
from recserve.serving.store import decode, encode


def test_roundtrip():
    rec = decode(
        {
            k.encode(): v
            for k, v in encode(
                np.array([5, 9]), np.array([1, 5, 9]), np.arange(len(USER_STATS))
            ).items()
        }
    )
    assert rec.recent.tolist() == [5, 9] and rec.seen.tolist() == [1, 5, 9]
    assert rec.stats.tolist() == list(range(len(USER_STATS)))
    assert rec.recent.dtype == np.int64


def test_missing_user_and_bad_stats():
    assert decode({}) is None
    with pytest.raises(ValueError):
        decode({k.encode(): v for k, v in encode([], [], np.zeros(2)).items()})
