import json
import os

import numpy as np

from train.metrics_io import (RESULTS_DIR, TRAIN_FIELDS, AppendCSV,
                              new_run_id, timestamp)


class TrainingMetrics:
    def __init__(self, save_path='training_metrics.json', run_id=None,
                 csv_path=None, config=None):
        self.save_path = save_path
        self.data = {
            'episodes': [],
            'agent_0': [],
            'agent_1': [],
            'actor_loss_avg': [],
            'critic_loss': [],
            'reward_avg': []
        }

        #CARGAR lo que ya hubiera
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    previous = json.load(f)
                for key in self.data:
                    if isinstance(previous.get(key), list):
                        self.data[key] = previous[key]
                print(f"[metrics] historico recuperado: "
                      f"{len(self.data['episodes'])} episodios de {save_path}")
            except Exception as e:
                #un json corrupto no puede tumbar el entrenamiento
                print(f"[metrics] no se pudo leer {save_path} ({e}), se empieza vacio")

        self.run_id = run_id or new_run_id("train")
        self.config = config or {}
        self._csv = AppendCSV(
            csv_path or os.path.join(RESULTS_DIR, "train_episodes.csv"),
            TRAIN_FIELDS)

    def log_episode(self, episode, actor_loss_dict, critic_loss_list, reward_avg,
                    per_agent=None):
        """per_agent: {agent_id: {reward, goals, collisions, offroad, timeouts,
        mean_velocity, route_completion}}"""
        self.data['episodes'].append(episode)

        agent_0_loss = np.mean(actor_loss_dict['agent_0']) if 'agent_0' in actor_loss_dict else 0
        agent_1_loss = np.mean(actor_loss_dict['agent_1']) if 'agent_1' in actor_loss_dict else 0

        self.data['agent_0'].append(float(agent_0_loss))
        self.data['agent_1'].append(float(agent_1_loss))
        self.data['actor_loss_avg'].append(float((agent_0_loss + agent_1_loss) / 2))

        critic_avg = np.mean(critic_loss_list) if critic_loss_list else 0
        self.data['critic_loss'].append(float(critic_avg))

        self.data['reward_avg'].append(float(reward_avg))

        self.save()

        #una fila por agente y episodio
        if per_agent:
            for agent_id, stats in per_agent.items():
                losses = actor_loss_dict.get(agent_id, [])
                row = {
                    "run_id": self.run_id,
                    "timestamp": timestamp(),
                    "episode": episode,
                    "agent_id": agent_id,
                    "reward": stats.get("reward", 0.0),
                    "avg_reward": reward_avg,
                    "actor_loss": float(np.mean(losses)) if len(losses) else 0.0,
                    "critic_loss": float(critic_avg),
                    "goals": stats.get("goals", 0),
                    "collisions": stats.get("collisions", 0),
                    "offroad": stats.get("offroad", 0),
                    "timeouts": stats.get("timeouts", 0),
                    "mean_velocity": stats.get("mean_velocity", 0.0),
                    "route_completion": stats.get("route_completion", 0.0),
                }
                row.update(self.config)
                self._csv.write(row)

    def save(self):
        #si el proceso muere a mitad del dump, el json
        #original sigue intacto en vez de quedarse truncado
        tmp = self.save_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, self.save_path)

    def close(self):
        self._csv.close()
