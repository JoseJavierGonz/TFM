import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
from pynput import keyboard
from algorithms.MAPPO import MAPPO, BufferExp
from env.gymCARLA import envCARLA
from train.metrics_train import TrainingMetrics


gamma = 0.99
lambda_var = 0.95
num_episodes = 300
num_agents = 2
rollout_steps = 2048  
best_reward = -float('inf')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = TrainingMetrics()
env = envCARLA()
mappo = MAPPO(num_agents=num_agents, space_obs=11, space_act=2, gamma=gamma, par_lambda=lambda_var, device=device)
listener = keyboard.Listener(on_press=env.CARLA.which_camera)
listener.start()
for episode in range(num_episodes):
    obs = env.reset()
    buffer = BufferExp()
    episode_rewards = {f"agent_{i}": 0 for i in range(num_agents)}
    
    # Limpiar memoria antes de empezar
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
        buffer_action = {}
        images_dict = {}
        
        for agent_idx in range(num_agents):
            agent_id = f"agent_{agent_idx}"
            state = torch.tensor(obs[agent_id]["vehicle_state"], dtype=torch.float32).unsqueeze(0).to(device)
            states_dict[agent_id] = state
            image = torch.from_numpy(obs[agent_id]["camera"]).permute(2,0,1).float()
            image = image.unsqueeze(0).to(device)
            images_dict[agent_id] = image
                        
            action, log_prob, act_buffer = mappo.politic(state, agent_id, image)
            actions_dict[agent_id] = action
            buffer_action[agent_id] = act_buffer.detach().cpu()
            log_probs_dict[agent_id] = log_prob.detach().cpu()
            
            #Liberar tensores intermedios
            del state, image
        
        global_state = torch.cat([states_dict[f"agent_{i}"] for i in range(num_agents)], dim=1)
        value = mappo.critic_evaluation(global_state).detach().squeeze()
        
        actions_list = [actions_dict[f"agent_{i}"].squeeze(0).detach().cpu().numpy() for i in range(num_agents)]
        next_obs, rewards_dict, dones_dict, _ = env.step(actions_list)
        
        for agent_id in rewards_dict.keys():
            episode_rewards[agent_id] += rewards_dict[agent_id]
        
        buffer.store(buffer_action, log_probs_dict, rewards_dict, states_dict, images_dict,
                     global_state.detach().cpu(), dones_dict, value)
        
        #Liberar variables grandes después de almacenarlas en buffer
        del actions_dict, log_probs_dict, states_dict, buffer_action, global_state, value, images_dict
        
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
    metrics.log_episode(
        episode,
        losses['actor_losses'],
        losses['critic_losses'],
        avg_reward
    )
    print(f"Episode {episode+1}/{num_episodes} - Avg Reward: {avg_reward:.2f}")

    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save({
            'actors': [actor.state_dict() for actor in mappo.actors],
            'critic': mappo.critic.state_dict(),
            'episode': episode,
            'best_reward': best_reward
        }, 'checkpoints/best_model.pt')

print("end")
listener.stop()
env.close()

