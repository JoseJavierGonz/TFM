import torch
from algorithms.MAPPO import MAPPO, BufferExp
from env.gymCARLA import envCARLA



gamma = 0.99
lambda_var = 0.95
num_episodes = 500
num_agents = 2
rollout_steps = 1024  

env = envCARLA()
mappo = MAPPO(num_agents=num_agents, space_obs=9, space_act=3, gamma=gamma, par_lambda=lambda_var)

for episode in range(num_episodes):
    obs = env.reset()
    buffer = BufferExp()
    episode_rewards = {f"agent_{i}": 0 for i in range(num_agents)}
    
    for step in range(rollout_steps):
        actions_dict = {}
        log_probs_dict = {}
        states_dict = {}
        
        for agent_idx in range(num_agents):
            agent_id = f"agent_{agent_idx}"
            state = obs[agent_id]["vehicle_state"]  
            states_dict[agent_id] = state
            
            action, log_prob = mappo.politic(state, agent_id)
            actions_dict[agent_id] = action
            log_probs_dict[agent_id] = log_prob
        
        global_state = torch.cat([torch.tensor(states_dict[f"agent_{i}"], dtype=torch.float32) 
                                   for i in range(num_agents)])
        value = mappo.critic_evaluation(global_state)
        
        actions_list = [actions_dict[f"agent_{i}"].detach().numpy() for i in range(num_agents)]
        next_obs, rewards_dict, dones_dict, _ = env.step(actions_list)
        
        for agent_id in rewards_dict.keys():
            episode_rewards[agent_id] += rewards_dict[agent_id]
        
        buffer.store(actions_dict, log_probs_dict, rewards_dict, states_dict, 
                     global_state, dones_dict, value)
        
        obs = next_obs
        
        if any(dones_dict.values()):
            obs = env.reset()
    
    mappo.update(buffer)
    buffer.clear_buffer()
    
    avg_reward = sum(episode_rewards.values()) / num_agents
    print(f"Episode {episode+1}/{num_episodes} - Avg Reward: {avg_reward:.2f}")

env.close()

