import sys
import os
import psutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import gc
from pynput import keyboard
from algorithms.MAPPO import MAPPO, BufferExp
from env.gymCARLA import envCARLA
from train.metrics_train import TrainingMetrics
from train.metrics_io import append_run_metadata, timestamp



gamma = 0.99
lambda_var = 0.95
num_episodes = 1500
restart_carla = 3
num_agents = 2
rollout_steps = 2048  
DECAY_EPISODES = 30
DECAY_STEPS = DECAY_EPISODES * rollout_steps
best_reward = -float('inf')
oom=False
process = psutil.Process(os.getpid())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_memory():
    """Debuggeo por los porblemas encontrados en CARLA"""
    rss = process.memory_info().rss / 1024 **2
    print(f"RSS {rss:.1f} MB")
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read()) / 1024 **2
        print(f"cgroup {current:.1f} MB")

        with open("/sys/fs/cgroup/memory.events") as f:
            print("events")
            print(f.read())
    except Exception as e:
        print("mem group unavailable: {e}")


def save_checkpoint(path, episode, mappo, best_reward):
    """Guardamos el modelo y los pesos. 
    Evitamos así tener que reentrenar siempre desde cero."""
    check = {'episode': episode,
                'actors': [actor.state_dict() for actor in mappo.actors],
                'critic': mappo.critic.state_dict(),
                'actors_op': [optimizer.state_dict() for optimizer in mappo.actors_op],
                'critic_op': mappo.critic_op.state_dict(),
                'best_reward': best_reward,}
    torch.save(check, path)


def load_checkpoint(path, mappo):
    """Cargamos modelo y pesos guardados"""
    check = torch.load(path, map_location=device, weights_only=False)
    for actor, sd in zip(mappo.actors, check['actors']):
        actor.load_state_dict(sd)
    mappo.critic.load_state_dict(check['critic'])
    if 'actors_op' in check and 'critic_op' in check: #momentaneo
        for optimizer, state_dict in zip(mappo.actors_op, check['actors_op']):
            optimizer.load_state_dict(state_dict)
        mappo.critic_op.load_state_dict(check['critic_op'])
    else:
        print("antiguo")

    best_reward = check.get('best_reward', -float('inf'))
    start_episode = check.get('episode', -1)
    print(f"resumed from {path} at {start_episode}")

    return start_episode + 1, best_reward


CAUSE_TO_KEY = {"goal": "goals", "collision": "collisions",
                "offroad": "offroad", "timeout": "timeouts"}

#escenario del entrenamiento
TRAIN_NPCS = 20
TRAIN_WALKERS = 20
TRAIN_CURRICULUM = 0.7
#metricas a estudiar
metrics = TrainingMetrics(config={
    "n_npcs": TRAIN_NPCS, "n_walkers": TRAIN_WALKERS,
    "curriculum_prob": TRAIN_CURRICULUM, "rollout_steps": rollout_steps,
})
append_run_metadata({
    "run_id": metrics.run_id, "timestamp": timestamp(), "mode": "train",
    "scenario": "train_traffic", "n_npcs": TRAIN_NPCS,
    "n_walkers": TRAIN_WALKERS, "curriculum_prob": TRAIN_CURRICULUM,
    "gamma": gamma, "lambda": lambda_var, "rollout_steps": rollout_steps,
    "num_agents": num_agents, "decay_episodes": DECAY_EPISODES,
})


env = envCARLA(num_vehicles=TRAIN_NPCS, num_walkers=TRAIN_WALKERS,
               curriculum_prob=TRAIN_CURRICULUM)
mappo = MAPPO(num_agents=num_agents, space_obs=12, space_act=2, gamma=gamma, par_lambda=lambda_var, device=device)  
listener = keyboard.Listener(on_press=env.CARLA.which_camera)
listener.start()

start_episode = 0
best_reward = -float('inf')
checkpoint_path = None
for path in ('checkpoints/model_restart.pt', 'checkpoints/best_model.pt'):
    if os.path.exists(path) and (checkpoint_path is None or os.path.getmtime(path) > os.path.getmtime(checkpoint_path)):
        checkpoint_path = path
if checkpoint_path:
    start_episode, best_reward = load_checkpoint(checkpoint_path, mappo)


#BUCLE PRINCIPAL DE ENTRENAMIENTO
for episode in range(start_episode, num_episodes):
    curretn_global_step = episode * rollout_steps
    for actor in mappo.actors:
        actor.decay_throttle_bias(curretn_global_step, DECAY_STEPS)
    obs = env.reset()
    buffer = BufferExp()
    episode_rewards = {f"agent_{i}": 0 for i in range(num_agents)}

    ep_stats = {f"agent_{i}": {"goals": 0, "collisions": 0, "offroad": 0,
                               "timeouts": 0, "vel_sum": 0.0, "vel_n": 0,
                               "route_completion": 0.0}
                for i in range(num_agents)}
    
    #Limpiar memoria antes de empezar
    if episode > 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    for step in range(rollout_steps):
        same_position = False if step == 0 else True
        if step % 100 == 0:
            if env.CARLA.camera_mode == 0:
                spectator = env.CARLA.world.get_spectator()
                env.CARLA.map_view(spectator)
            elif env.CARLA.camera_mode == 1:
                env.CARLA.follow_vehicle(env.CARLA.vehicles_marl_list[0])
            elif env.CARLA.camera_mode == 2:
                env.CARLA.follow_vehicle(env.CARLA.vehicles_marl_list[1])

            print(f"  Step {step}/{rollout_steps}")

        actions_dict = {}
        log_probs_dict = {}
        states_dict = {}
        cam_dict = {}
        buffer_action = {}
        
        for agent_idx in range(num_agents):
            agent_id = f"agent_{agent_idx}"
            state = torch.tensor(obs[agent_id]["vehicle_state"], dtype=torch.float32).unsqueeze(0).to(device)
            cam_state = torch.tensor(obs[agent_id]["cam_features"], dtype=torch.float32).unsqueeze(0).to(device)
            states_dict[agent_id] = state
            cam_dict[agent_id] = cam_state

                        
            action, log_prob, act_buffer = mappo.politic(state, cam_state, agent_id)

            actions_dict[agent_id] = action
            buffer_action[agent_id] = act_buffer.detach().cpu()
            log_probs_dict[agent_id] = log_prob.detach().cpu()
            
            #Liberar tensores intermedios
            del state, cam_state
        
        global_state = torch.cat([states_dict[f"agent_{i}"] for i in range(num_agents)], dim=1)
        global_state_cam = torch.cat([cam_dict[f"agent_{i}"] for i in range(num_agents)], dim=1)
        value = mappo.critic_evaluation(global_state, global_state_cam).detach().squeeze()

        
        actions_list = [actions_dict[f"agent_{i}"].squeeze(0).detach().cpu().numpy() for i in range(num_agents)]
        next_obs, rewards_dict, dones_dict, info = env.step(actions_list)

        for agent_id, st in ep_stats.items():
            v = info["velocity"].get(agent_id)
            if v is not None:
                st["vel_sum"] += v
                st["vel_n"] += 1
            cause = info["termination"].get(agent_id)
            if cause:
                st[CAUSE_TO_KEY[cause]] += 1
                st["route_completion"] = info["route_completion"].get(agent_id, 0.0)

        
        for agent_id in rewards_dict.keys():
            episode_rewards[agent_id] += rewards_dict[agent_id]
        
        buffer.store(buffer_action, log_probs_dict, rewards_dict, states_dict, cam_dict,
                     global_state.detach().cpu(), global_state_cam.detach().cpu(), dones_dict, value)
        
        #Liberar variables grandes después de almacenarlas en buffer
        del actions_dict, log_probs_dict, states_dict, buffer_action, global_state, global_state_cam, cam_dict, value
        
        obs = next_obs
        
        agents_to_reset = [
            agent_id for agent_id, done in dones_dict.items()
            if done and agent_id != "__all__"
        ]
        if agents_to_reset:
            obs = env.reset(agent_ids=agents_to_reset, same_position=same_position)

    losses = mappo.update(buffer)
    buffer.clear_buffer()
    del buffer

    #Limpiar GPU y CPU después de cada episodio
    torch.cuda.empty_cache()
    gc.collect()
    
    avg_reward = sum(episode_rewards.values()) / num_agents
    per_agent = {}
    for agent_id, st in ep_stats.items():
        per_agent[agent_id] = {
            "reward": episode_rewards[agent_id],
            "goals": st["goals"], "collisions": st["collisions"],
            "offroad": st["offroad"], "timeouts": st["timeouts"],
            "mean_velocity": st["vel_sum"] / max(1, st["vel_n"]),
            "route_completion": st["route_completion"],
        }
    metrics.log_episode(
        episode,
        losses['actor_losses'],
        losses['critic_losses'],
        avg_reward,
        per_agent=per_agent
    )
    print(f"Episode {episode+1}/{num_episodes} - Avg Reward: {avg_reward:.2f}")

    if avg_reward > best_reward:
        best_reward = avg_reward
        save_checkpoint('checkpoints/best_model.pt',
                        episode,
                        mappo,
                        best_reward)

    #Solucion mas repetida en stack overflow y otras
    #reseteamos CARLA cada x episodios para evitar en todo lo posible fallos
    if (episode + 1) % restart_carla == 0 and (episode + 1) < num_episodes:
        print(f"Disconnect and connect from CARLA, episode {episode + 1}")

        save_checkpoint('checkpoints/model_restart.pt',
                        episode,
                        mappo,
                        best_reward)
        
        max_retries = 3

        for i in range(max_retries):
            try:
                listener.stop()
                env.close()
                del env
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)
                env = envCARLA(num_vehicles=TRAIN_NPCS, num_walkers=TRAIN_WALKERS,
                               curriculum_prob=TRAIN_CURRICULUM)
                listener = keyboard.Listener(on_press=env.CARLA.which_camera)
                listener.start()
                print("Reconnection completed")
                break
            except Exception as e:
                print(f"RESTART ERROR at close env: {e}. Attempt {i+1}")

                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)
        else:
            raise RuntimeError("CARLA can't be reached")

last_episode = num_episodes -1
save_checkpoint('checkpoints/final_model.pt',
                last_episode,
                mappo,
                best_reward)      

print("end")
metrics.close()
listener.stop()
listener.join()
env.close()

