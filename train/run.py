"""
Ejecutamos episodios sin exploración y obtenemos las métricas 
que nos dirán como de bueno es nuestro modelo tras entrenar.

Ejemplos
--------
    python run.py --checkpoint checkpoints/model_restart.pt --episodes 20
    python run.py --stochastic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import torch
from torch.distributions import Normal
from pynput import keyboard

from env.gymCARLA import envCARLA
from models.networks import Actor_network
from train.metrics_io import (RESULTS_DIR, EVAL_FIELDS, AppendCSV,
                              append_run_metadata, new_run_id, timestamp)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MAPPO/CARLA checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                   help="Checkpoint saved by train_custom.py")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2200,
                   help="Safety cap per episode (env.max_steps is 2050)")
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--space-obs", type=int, default=12) 
    p.add_argument("--space-act", type=int, default=2)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from the policy instead of taking the mean action")
    p.add_argument("--follow", action="store_true",
                   help="Move the spectator camera during evaluation (press 0/1/2 to switch, like training)")
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    p.add_argument("--output", default=None, help="Optional path to dump a JSON summary")
    p.add_argument("--scenario", default="traffic",
                   help="Nombre del escenario, va en cada fila del CSV (p.ej. no_traffic)")
    p.add_argument("--npcs", type=int, default=20, help="Vehiculos NPC (0 = mapa vacio)")
    p.add_argument("--walkers", type=int, default=20, help="Peatones (0 = ninguno)")
    p.add_argument("--curriculum-prob", type=float, default=0.0,
                   help="0 en evaluacion: el curriculum es una ayuda de entrenamiento")
    p.add_argument("--csv", default=None, help="CSV append-only (por defecto results/eval_episodes.csv)")
    return p.parse_args()


def load_actors(checkpoint_path, num_agents, space_obs, space_act, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    actors = [Actor_network(space_obs, space_act).to(device) for _ in range(num_agents)]
    for i, actor in enumerate(actors):
        actor.load_state_dict(checkpoint["actors"][i])
        actor.eval()

    print(f"Loaded '{checkpoint_path}' "
          f"(episode={checkpoint.get('episode')}, best_reward={checkpoint.get('best_reward'):.2f})")
    return actors


@torch.no_grad()
def select_action(actor, state, cam_state, stochastic):
    """Acciones durante la evaluacion sin ruido, sacamos solo la media
    Se puede meter --stochastic y comprobar diferencias."""
    mean, std = actor(state, cam_state)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                print(f"[WARN] NaN/Inf en la politica, usando accion neutra")
                mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
                std = torch.nan_to_num(std, nan=0.1, posinf=0.1, neginf=0.1).clamp(min=1e-3)
    raw = Normal(mean, std).sample() if stochastic else mean
    throttle = torch.tanh(raw[:, 0:1])
    steer = torch.tanh(raw[:, 1:2])
    return torch.cat([throttle, steer], dim=1)


def run_episode(env, actors, agent_ids, device, max_steps, stochastic, follow):
    obs = env.reset()
    reward_sum = {aid: 0.0 for aid in agent_ids}
    steps_alive = {aid: 0 for aid in agent_ids}
    outcome = {aid: "timeout" for aid in agent_ids}
    finished = {aid: False for aid in agent_ids}
    extra = {aid: {"vel_sum": 0.0, "vel_n": 0, "max_velocity": 0.0,
                   "dist_to_goal_final": 0.0, "route_completion": 0.0,
                   "initial_dist": 0.0} for aid in agent_ids}

    for step in range(max_steps):
        if follow and step % 20 == 0:
            spectator = env.CARLA.world.get_spectator()
            if env.CARLA.camera_mode == 0:
                env.CARLA.map_view(spectator)
            elif env.CARLA.camera_mode == 1:
                env.CARLA.follow_vehicle(env.CARLA.vehicles_marl_list[0])
            elif env.CARLA.camera_mode == 2:
                env.CARLA.follow_vehicle(env.CARLA.vehicles_marl_list[1])

        actions_list = []
        for idx, agent_id in enumerate(agent_ids):
            state = torch.tensor(obs[agent_id]["vehicle_state"], dtype=torch.float32).unsqueeze(0).to(device)
            cam_state = torch.tensor(obs[agent_id]["cam_features"], dtype=torch.float32).unsqueeze(0).to(device)
            action = select_action(actors[idx], state, cam_state, stochastic)
            actions_list.append(action.squeeze(0).cpu().numpy())

        obs, rewards, dones, info = env.step(actions_list)

        for idx, agent_id in enumerate(agent_ids):
            if finished[agent_id]:
                continue
            reward_sum[agent_id] += rewards[agent_id]
            steps_alive[agent_id] += 1
            v = info["velocity"].get(agent_id, 0.0)
            extra[agent_id]["vel_sum"] += v
            extra[agent_id]["vel_n"] += 1
            extra[agent_id]["max_velocity"] = max(extra[agent_id]["max_velocity"], v)
            if dones[agent_id]:
                finished[agent_id] = True
                outcome[agent_id] = info["termination"].get(agent_id) or "timeout"
                extra[agent_id]["dist_to_goal_final"] = info["dist_to_goal"].get(agent_id, 0.0)
                extra[agent_id]["route_completion"] = info["route_completion"].get(agent_id, 0.0)
                extra[agent_id]["initial_dist"] = info["initial_dist"].get(agent_id, 0.0)

        if all(finished.values()):
            break

    return reward_sum, steps_alive, outcome, extra


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    actors = load_actors(args.checkpoint, args.num_agents, args.space_obs, args.space_act, device)

    env = envCARLA(num_vehicles=args.npcs, num_walkers=args.walkers,
                   curriculum_prob=args.curriculum_prob)
    agent_ids = env.agent_id

    run_id = new_run_id(f"eval_{args.scenario}")
    scenario_cfg = {"n_npcs": args.npcs, "n_walkers": args.walkers,
                    "curriculum_prob": args.curriculum_prob,
                    "deterministic": not args.stochastic}
    append_run_metadata({"run_id": run_id, "timestamp": timestamp(), "mode": "eval",
                         "scenario": args.scenario, "checkpoint": args.checkpoint,
                         "episodes": args.episodes, "max_steps": args.max_steps,
                         **scenario_cfg})
    csv_log = AppendCSV(args.csv or os.path.join(RESULTS_DIR, "eval_episodes.csv"),
                        EVAL_FIELDS)

    listener = None
    if args.follow:
        listener = keyboard.Listener(on_press=env.CARLA.which_camera)
        listener.start()

    all_results = {aid: {"reward": [], "steps": [], "outcome": []} for aid in agent_ids}

    try:
        for ep in range(args.episodes):
            reward_sum, steps_alive, outcome, extra = run_episode(
                env, actors, agent_ids, device, args.max_steps, args.stochastic, args.follow
            )
            summary = " | ".join(
                f"{aid}: {outcome[aid]:<9} reward={reward_sum[aid]:7.2f} steps={steps_alive[aid]}"
                for aid in agent_ids
            )
            print(f"Episode {ep + 1}/{args.episodes}  {summary}")

            #una fila por agente, si CARLA se cae en el
            #episodio siguiente no se pierde nada 
            for aid in agent_ids:
                all_results[aid]["reward"].append(reward_sum[aid])
                all_results[aid]["steps"].append(steps_alive[aid])
                all_results[aid]["outcome"].append(outcome[aid])
                e = extra[aid]
                csv_log.write({
                    "run_id": run_id, "timestamp": timestamp(),
                    "scenario": args.scenario, "checkpoint": args.checkpoint,
                    "episode": ep, "agent_id": aid, "outcome": outcome[aid],
                    "reward": reward_sum[aid], "steps_alive": steps_alive[aid],
                    "mean_velocity": e["vel_sum"] / max(1, e["vel_n"]),
                    "max_velocity": e["max_velocity"],
                    "dist_to_goal_final": e["dist_to_goal_final"],
                    "route_completion": e["route_completion"],
                    "initial_dist": e["initial_dist"],
                    **scenario_cfg,
                })
    finally:
        if listener is not None:
            listener.stop()
        csv_log.close()
        env.close()

    print(f"\n=== Summary over {args.episodes} episodes ===")
    final_summary = {}
    for aid in agent_ids:
        outcomes = all_results[aid]["outcome"]
        n = len(outcomes)
        stats = {
            "goal_rate": outcomes.count("goal") / n,
            "collision_rate": outcomes.count("collision") / n,
            "timeout_rate": outcomes.count("timeout") / n,
            "offroad_rate": outcomes.count("offroad") / n,
            "avg_reward": float(np.mean(all_results[aid]["reward"])),
            "avg_steps_alive": float(np.mean(all_results[aid]["steps"])),
        }
        final_summary[aid] = stats
        print(f"{aid}: goal={stats['goal_rate']:.0%}  collision={stats['collision_rate']:.0%}  "
              f"timeout={stats['timeout_rate']:.0%}  offroad={stats['offroad_rate']:.0%}  "
              f"avg_reward={stats['avg_reward']:.2f}  "
              f"avg_steps_alive={stats['avg_steps_alive']:.1f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
