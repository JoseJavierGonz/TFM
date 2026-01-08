import torch
from torch.distributions import Normal
from torch.optim import Adam
from models.networks import Actor_network, Critic_Actor


class MAPPO:
    def __init__(self, num_agents, space_obs, space_act, gamma, par_lambda):
        self.num_agents = num_agents
        self.agent_id_to_idx = {f"agent_{i}": i for i in range(num_agents)}
     
        self.actors = [Actor_network(space_obs, space_act) for _ in range(num_agents)]
        self.critic = Critic_Actor(space_obs * num_agents)  

        self.actors_op = [Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_op = Adam(self.critic.parameters(), lr=3e-4)
        self.gamma = gamma
        self.par_lambda = par_lambda


    def politic(self, state, agent_id):
        if isinstance(agent_id, str):
            actor_id = self.agent_id_to_idx[agent_id]
        else:
            actor_id = agent_id

        actor = self.actors[actor_id]
        mean, std = actor.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()

        prob = dist.log_prob(action).sum()

        return action, prob
    
    def critic_evaluation(self, state_final):
        return self.critic(state_final)
    

    def update(self, buffer):
        advantages = []
        gae = 0
        for t in reversed(range(len(buffer.global_states))):
            if t == len(buffer.global_states) -1 :
                next_val = 0
            else:
                any_done = any(buffer.dones[agent_id][t] for agent_id in buffer.dones.keys())
                next_val = 0 if any_done else buffer.critic_values[t+1]
            
            
            mean_reward = sum(buffer.rewards[agent_id][t] for agent_id in buffer.rewards.keys()) / len(buffer.rewards)

            g_t = mean_reward + self.gamma * next_val - buffer.critic_values[t]
            gae = g_t + self.gamma * self.par_lambda * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for epoch in range(10):
            for agent_idx, (agent_id, actor) in enumerate(zip(buffer.actions.keys(), self.actors)):
                state_list = buffer.states[agent_id]
                old_actions_list = buffer.actions[agent_id]
                old_probs_list = buffer.log_probs[agent_id]
                
                states_tensor = torch.stack([torch.tensor(stat, dtype=torch.float32) for stat in state_list])
                actions_tensor = torch.stack([torch.tensor(act, dtype=torch.float32) for act in old_actions_list])
                old_log_probs_tensor = torch.stack([torch.tensor(prob, dtype=torch.float32) for prob in old_probs_list])
            
                mean, std = actor(states_tensor)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)

                ratio = torch.exp(new_log_probs - old_log_probs_tensor)

                reinforce = ratio * advantages
                clipping = torch.clamp(ratio, 0.8, 1.2) * advantages
                actor_loss = -torch.min(reinforce, clipping).mean()

                self.actor_op[agent_idx].zero_grad()
                actor_loss.backward()
                self.actor_op[agent_idx].step()
            
            global_state_tensor = torch.stack([torch.tensor(gs, dtype=torch.float32) for gs in buffer.global_states])
            predicted_values = self.critic(global_state_tensor).squeeze()


            target_values = advantages + torch.tensor(buffer.critic_values, dtype=torch.float32)
            
            critic_loss = (predicted_values - target_values).pow(2).mean()
            
            self.critic_op.zero_grad()
            critic_loss.backward()
            self.critic_op.step()




class BufferExp:
    def __init__(self):
        self.actions = {}
        self.log_probs = {}
        self.rewards = {}
        self.states = {}
        self.dones = {}
        self.critic_values = []
        self.global_states = []

    def store(self, actions_dict, log_probs_dict, rewards_dict, states_dict, 
              global_state, dones_dict, value):
        for agent_id in actions_dict.keys():
            if agent_id not in self.actions:
                self.actions[agent_id] = []
                self.log_probs[agent_id] = []
                self.rewards[agent_id] = []
                self.states[agent_id] = []
                self.dones[agent_id] = []
        
        for agent_id in actions_dict.keys():
            self.actions[agent_id].append(actions_dict[agent_id])
            self.log_probs[agent_id].append(log_probs_dict[agent_id])
            self.rewards[agent_id].append(rewards_dict[agent_id])
            self.states[agent_id].append(states_dict[agent_id])
            self.dones[agent_id].append(dones_dict[agent_id])
        
        self.global_states.append(global_state)
        self.critic_values.append(value)

    def clear_buffer(self):
        self.__init__()
        
    def __len__(self):
        return len(self.states)



