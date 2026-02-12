import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pynput import keyboard
from algorithms.MAPPO import MAPPO, BufferExp
from env.gymCARLA import envCARLA



gamma = 0.99
expl_coef = 0.99
lambda_var = 0.95
num_episodes = 500
num_agents = 2
rollout_steps = 1024  
best_reward = -float('inf')

env = envCARLA()
mappo = MAPPO(num_agents=num_agents, space_obs=10, space_act=3, gamma=gamma, par_lambda=lambda_var)
listener = keyboard.Listener(on_press=env.CARLA.which_camera)
listener.start()
for episode in range(num_episodes):
    obs = env.reset()
    buffer = BufferExp()
    episode_rewards = {f"agent_{i}": 0 for i in range(num_agents)}
    
    for step in range(rollout_steps):
        same_position = False if step == 0 else True
        if step % 20 == 0:
            if env.CARLA.camera_mode == 0:
                spectator = env.CARLA.world.get_spectator()
                env.CARLA.map_view(spectator)
            elif env.CARLA.camera_mode == 1:
                env.CARLA.follow_vehicle(env.CARLA.vehicles_marl_list[0])
            elif env.CARLA.camera_mode == 2:
                env.CARLA.follow_vehicle(env.CARLA.vehicles_marl_list[1])
    
        if step % 100 == 0:
            print(f"  Step {step}/{rollout_steps}")
        actions_dict = {}
        log_probs_dict = {}
        states_dict = {}
        
        for agent_idx in range(num_agents):
            agent_id = f"agent_{agent_idx}"
            state = torch.tensor(obs[agent_id]["vehicle_state"], dtype=torch.float32).unsqueeze(0)
            states_dict[agent_id] = state
            
            
            action, log_prob = mappo.politic(state, agent_id, expl_coef)
            actions_dict[agent_id] = action
            log_probs_dict[agent_id] = log_prob
        
        global_state = torch.cat([states_dict[f"agent_{i}"].detach() for i in range(num_agents)], dim=1)
        value = mappo.critic_evaluation(global_state).detach()
        
        actions_list = [actions_dict[f"agent_{i}"].squeeze(0).detach().numpy() for i in range(num_agents)]
        next_obs, rewards_dict, dones_dict, _ = env.step(actions_list)
        
        for agent_id in rewards_dict.keys():
            episode_rewards[agent_id] += rewards_dict[agent_id]
        
        buffer.store(actions_dict, log_probs_dict, rewards_dict, states_dict, 
                     global_state, dones_dict, value)
        
        obs = next_obs
        
        agents_to_reset = [agent_id for agent_id, done in dones_dict.items() 
                          if done and agent_id != "__all__"]
        if agents_to_reset:
            obs = env.reset(agent_ids=agents_to_reset, same_position=same_position)
    
    mappo.update(buffer)
    buffer.clear_buffer()
    
    avg_reward = sum(episode_rewards.values()) / num_agents
    print(f"Episode {episode+1}/{num_episodes} - Avg Reward: {avg_reward:.2f}")
    expl_coef = max(0.05, 0.99- episode/num_episodes)

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

