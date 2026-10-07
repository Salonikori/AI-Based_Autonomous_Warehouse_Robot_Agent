"""
Machine-learning decision layer for the Warehouse Robot Agent.

The planner remains safety-critical and symbolic (A* + battery constraints).
This module adds a learned policy that estimates how valuable it is to serve
each feasible order *now*. The model is trained deterministically from
synthetic warehouse experiences so the project runs without an external API,
LLM key, or cloud service.

Model: RandomForestRegressor
Features:
    priority, waiting_time, distance_to_pickup, delivery_distance,
    battery, trip_energy, obstacle_count, urgency
Target:
    learned service value (higher = serve sooner)

This is a hybrid AI agent: ML selects the task, while hard constraints and
A* guarantee that the selected action remains physically/planning feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Iterable

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


FEATURE_NAMES = (
    "priority",
    "waiting_time",
    "distance_to_pickup",
    "delivery_distance",
    "battery",
    "trip_energy",
    "obstacle_count",
    "urgency",
)


@dataclass
class AIPrediction:
    task_id: int
    score: float
    confidence: float
    reason: str


def _features(priority, waiting_time, distance_to_pickup,
               delivery_distance, battery, trip_energy, obstacle_count):
    urgency = priority * (1.0 + min(waiting_time, 60) / 60.0)
    return [
        float(priority),
        float(waiting_time),
        float(distance_to_pickup),
        float(delivery_distance),
        float(battery),
        float(trip_energy),
        float(obstacle_count),
        float(urgency),
    ]


def _synthetic_dataset(n=5000, seed=42):
    """Create reproducible training examples from warehouse decision signals."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        priority = rng.choice([1, 1, 1, 2, 3])
        waiting = rng.randint(0, 120)
        pickup = rng.randint(1, 35)
        delivery = rng.randint(5, 35)
        battery = rng.randint(15, 100)
        trip = pickup + delivery + 10
        obstacles = rng.randint(0, 6)

        urgency = priority * (1 + min(waiting, 60) / 60)
        # A nonlinear expert-policy signal used only to generate training
        # experience. The ML model learns this relationship from examples.
        score = (
            2.8 * urgency
            + 7.0 / (pickup + 1)
            + 5.0 / (delivery + 1)
            + 0.035 * waiting
            + 0.012 * battery
            - 0.055 * trip
            - 0.30 * obstacles
            + (1.8 if priority == 3 and waiting > 8 else 0)
            + rng.gauss(0, 0.45)
        )
        X.append(_features(priority, waiting, pickup, delivery,
                           battery, trip, obstacles))
        y.append(score)
    return X, y


@lru_cache(maxsize=1)
def get_model():
    if not SKLEARN_AVAILABLE:
        return None

    X, y = _synthetic_dataset()
    model = Pipeline([
        ("scale", StandardScaler()),
        ("forest", RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X, y)
    return model


class OrderDecisionModel:
    """ML policy used by the robot for autonomous order selection."""

    name = "Warehouse Order Policy"
    algorithm = "Random Forest Regressor"
    version = "1.0"

    def __init__(self):
        self.model = get_model()
        self.enabled = self.model is not None

    @property
    def status(self):
        return "online" if self.enabled else "fallback"

    def predict(self, task, pos, battery, trip_energy, obstacle_count, t):
        waiting = max(0, t - task.created)
        pickup_distance = abs(pos[0] - task.pickup[0]) + abs(pos[1] - task.pickup[1])
        delivery_distance = abs(task.pickup[0] - task.drop[0]) + abs(task.pickup[1] - task.drop[1])
        features = _features(
            task.priority, waiting, pickup_distance, delivery_distance,
            battery, trip_energy, obstacle_count,
        )

        if self.model is not None:
            score = float(self.model.predict([features])[0])
        else:
            # Safe deterministic fallback if sklearn is unavailable.
            score = (
                task.priority * 10
                + 0.2 * waiting
            ) / (pickup_distance + delivery_distance + 1)

        # Confidence is a transparent relative confidence, not a probability.
        # It expresses how much this task's score stands out in this queue.
        return score

    def rank(self, tasks: Iterable, agent, t):
        tasks = list(tasks)
        if not tasks:
            return []

        raw = []
        for task in tasks:
            energy = agent.energy_needed(task)
            score = self.predict(
                task, agent.pos, agent.battery, energy,
                len(agent.blocked), t
            )
            raw.append((task, score))

        values = [s for _, s in raw]
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-9)

        ranked = []
        for task, score in sorted(raw, key=lambda x: x[1], reverse=True):
            confidence = 0.55 + 0.40 * ((score - lo) / span)
            reason = (
                f"P{task.priority}, waiting {max(0, t-task.created)} ticks, "
                f"ML score {score:.2f}"
            )
            ranked.append(AIPrediction(
                task_id=task.tid,
                score=score,
                confidence=confidence,
                reason=reason,
            ))
        return ranked


def model_info():
    return {
        "name": OrderDecisionModel.name,
        "algorithm": OrderDecisionModel.algorithm,
        "version": OrderDecisionModel.version,
        "status": "online" if SKLEARN_AVAILABLE else "fallback",
        "features": list(FEATURE_NAMES),
        "description": (
            "Learns order-selection value from warehouse decision examples. "
            "Hard battery constraints and A* remain safety guards."
        ),
    }
