"""Escritura de metricas tolerante a fallos.

CARLA se cae. El proceso se mata. La GPU se queda sin memoria. Todo lo que se
escriba aqui tiene que sobrevivir a un `kill -9` en mitad del entrenamiento, asi
que cada fila se vuelca a disco en el momento (flush + fsync) y NUNCA se abre un
fichero en modo 'w': solo se anade.
"""
import csv
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def new_run_id(prefix="run"):
    """Identificador unico por ejecucion: permite que varias convivan en el
    mismo CSV y separarlas luego facilmente."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def timestamp():
    return datetime.now().isoformat(timespec="seconds")


class AppendCSV:
    """CSV que solo añade. Asi al reanudar tras un fallo continua el mismo
    fichero en vez de machacarlo."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames,
                                 extrasaction="ignore")
        if need_header:
            self._w.writeheader()
            self._sync()

    def write(self, row):
        self._w.writerow(row)
        self._sync()

    def _sync(self):
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def append_run_metadata(meta, path=None):
    path = path or os.path.join(RESULTS_DIR, "runs.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, default=str) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


TRAIN_FIELDS = [
    "run_id", "timestamp", "episode", "agent_id",
    "reward", "avg_reward", "actor_loss", "critic_loss",
    "goals", "collisions", "offroad", "timeouts",
    "mean_velocity", "route_completion",
    "n_npcs", "n_walkers", "curriculum_prob", "rollout_steps",
]

EVAL_FIELDS = [
    "run_id", "timestamp", "scenario", "checkpoint", "episode", "agent_id",
    "outcome", "reward", "steps_alive",
    "mean_velocity", "max_velocity", "dist_to_goal_final",
    "route_completion", "initial_dist",
    "n_npcs", "n_walkers", "curriculum_prob", "deterministic",
]
