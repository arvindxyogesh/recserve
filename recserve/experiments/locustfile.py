"""Locust user for the recserve API, driven by recserve.experiments.load_test.

Each simulated user sends GET /recommend at a constant rate (RECSERVE_USER_RPS),
starting at a random phase,
for a random userId drawn from RECSERVE_USER_IDS (a .npy file), so the
offered load is users x per-user rate regardless of response time, up to
the point where the server can no longer keep up.
"""

from __future__ import annotations

import os
import random

import gevent
import numpy as np
from locust import constant_throughput, task
from locust.contrib.fasthttp import FastHttpUser

USER_IDS = np.load(os.environ["RECSERVE_USER_IDS"]).tolist()
PER_USER_RPS = float(os.environ.get("RECSERVE_USER_RPS", "5"))
K = int(os.environ.get("RECSERVE_K", "10"))


class RecommendUser(FastHttpUser):
    wait_time = constant_throughput(PER_USER_RPS)

    def on_start(self) -> None:
        # Random phase within one period: without it the users fire in lockstep
        # and the server sees synchronised bursts instead of a steady stream.
        gevent.sleep(random.uniform(0, 1 / PER_USER_RPS))

    @task
    def recommend(self) -> None:
        uid = random.choice(USER_IDS)
        self.client.get(f"/recommend?user_id={uid}&k={K}", name="/recommend")
