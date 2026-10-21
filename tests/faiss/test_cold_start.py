import json

import numpy as np

from recserve.serving.build_artifacts import cold_start_lists


def test_cold_start_lists_drop_zero_popularity_and_split_by_genre():
    pop = np.array([0, 5, 3, 9.0])
    genres = np.array([[1], [1], [0], [0.0]])
    lists = cold_start_lists(pop, genres, ["a"], 2)
    assert lists == {"global": [3, 1], "by_genre": {"a": [1]}}
    json.dumps(lists)
