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
        action_to_buffer = dist.sample()
        
        throttle = torch.tanh(action_to_buffer[:, 0:1])
        steer = torch.tanh(action_to_buffer[:, 1:2])
        action_tensor = torch.cat([throttle, steer], dim=1)

        prob = dist.log_prob(action_to_buffer).sum(dim=-1)
        

        return action_tensor, prob, action_to_buffer
    
    def critic_evaluation(self, state_final):
        return self.critic(state_final)
    

    def update(self, buffer):

        #PARA ACTUALIZAR LOS AGENTES
        advantages = {}
        len_global = len(buffer.global_states)
        for agent_id in buffer.rewards.keys():
            agent_adv=[]
            gae = 0
            for t in reversed(range(len_global)):
                if t == len_global -1 :
                    next_val = 0
                else:
                    next_val = 0 if buffer.dones[agent_id][t] else buffer.critic_values[t+1]
                
                reward = buffer.rewards[agent_id][t]
                g_t = reward + self.gamma * next_val - buffer.critic_values[t]
                gae = g_t + self.gamma * self.par_lambda * gae
                agent_adv.insert(0, gae)


            advi = torch.tensor(agent_adv, dtype=torch.float32)
            advantages[agent_id] = (advi- advi.mean()) / (advi.std()+ 1e-7)

        #PARA ACTUALIZAR EL CRITIC
        rewards_global = []
        for t in range(len_global):
            mean_reward_t = sum(buffer.rewards[agent][t] for agent in buffer.rewards.keys()) / self.num_agents
            rewards_global.append(mean_reward_t)

        advantages_global = []
        gae_g = 0.0

        for t in reversed(range(len_global)):
            if t == len_global -1 :
                next_val = 0
            else:
                next_val = 0 if any(buffer.dones[agent][t] for agent in buffer.dones.keys()) else buffer.critic_values[t+1]
            
            g_t = rewards_global[t] + self.gamma * next_val - buffer.critic_values[t]
            gae_g = g_t + self.gamma * self.par_lambda * gae_g
            advantages_global.insert(0, gae_g)


        advantages_global = torch.tensor(advantages_global, dtype=torch.float32)
        target_values = advantages_global + torch.tensor(buffer.critic_values, dtype=torch.float32) 

        losses_log = {
            'actor_losses': {agent_id: [] for agent_id in buffer.actions.keys()},
            'critic_losses': []
        }

        for epoch in range(5):
            for agent_idx, (agent_id, actor) in enumerate(zip(buffer.actions.keys(), self.actors)):
                state_list = buffer.states[agent_id]
                old_actions_list = buffer.actions[agent_id]
                old_probs_list = buffer.log_probs[agent_id]
                
                states_tensor = torch.stack([torch.tensor(stat, dtype=torch.float32) for stat in state_list]).squeeze(1)
                actions_tensor = torch.stack([torch.tensor(act, dtype=torch.float32) for act in old_actions_list]).squeeze(1)
                old_log_probs_tensor = torch.stack([torch.tensor(prob, dtype=torch.float32) for prob in old_probs_list]).squeeze(-1)
            
                mean, std = actor(states_tensor)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
                
                # Entropy para incentivar exploración
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_tensor)

                reinforce = ratio * advantages[agent_id]
                clipping = torch.clamp(ratio, 0.8, 1.2) * advantages[agent_id]
                actor_loss = -torch.min(reinforce, clipping).mean() - 0.05 * entropy
                losses_log['actor_losses'][agent_id].append(actor_loss.item())

                self.actors_op[agent_idx].zero_grad()
                actor_loss.backward()
            
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                self.actors_op[agent_idx].step()
            
            global_state_tensor = torch.stack([torch.tensor(gs, dtype=torch.float32) for gs in buffer.global_states]).squeeze(1)
            predicted_values = self.critic(global_state_tensor).squeeze(-1)

            
            critic_loss = (predicted_values - target_values).pow(2).mean()
            losses_log['critic_losses'].append(critic_loss.item())
            
            self.critic_op.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_op.step()

        return losses_log


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
            self.actions[agent_id].append(actions_dict[agent_id].detach())
            self.log_probs[agent_id].append(log_probs_dict[agent_id].detach())
            self.rewards[agent_id].append(rewards_dict[agent_id])
            self.states[agent_id].append(states_dict[agent_id].detach())
            self.dones[agent_id].append(dones_dict[agent_id])
        
        self.global_states.append(global_state)
        self.critic_values.append(value.item())

    def clear_buffer(self):
        self.__init__()
        
    def __len__(self):
        return len(self.states)



