import torch
from torch.distributions import Normal
from torch.optim import Adam
from models.networks import Actor_network, Critic_Actor


class MAPPO:
    def __init__(self, num_agents, space_obs, space_act, gamma, par_lambda, device):
        self.device = device
        self.num_agents = num_agents
        self.agent_id_to_idx = {f"agent_{i}": i for i in range(num_agents)}
     
        self.actors = [Actor_network(space_obs, space_act).to(self.device) for _ in range(num_agents)]
        self.critic = Critic_Actor(space_obs * num_agents).to(self.device)

        self.actors_op = [Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_op = Adam(self.critic.parameters(), lr=3e-4)
        self.gamma = gamma
        self.par_lambda = par_lambda


    def politic(self, state, agent_id, image):
        if isinstance(agent_id, str):
            actor_id = self.agent_id_to_idx[agent_id]
        else:
            actor_id = agent_id

        actor = self.actors[actor_id]
        mean, std = actor(image, state)
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
        advantages = {}
        targets = {}
        values = torch.stack(buffer.critic_values).squeeze(-1).to(self.device).detach()
        len_global = len(buffer.global_states)

        for agent_id in buffer.rewards.keys():
            agent_adv = []
            gae = 0
            for t in reversed(range(len_global)):
                next_val = 0 if (t == len_global - 1 or buffer.dones[agent_id][t]) else values[t+1]
                reward = buffer.rewards[agent_id][t]
                delta = reward + self.gamma * next_val - values[t]
                gae = delta + self.gamma * self.par_lambda * gae
                agent_adv.insert(0, gae)

            advi = torch.tensor(agent_adv, dtype=torch.float32).to(self.device)
            targets[agent_id] = advi + values
            advantages[agent_id] = (advi - advi.mean()) / (advi.std() + 1e-7)

        target_values = torch.stack([targets[aid] for aid in targets]).mean(dim=0).detach()

        precomputed = {}
        for agent_idx in range(self.num_agents):
            agent_id = f"agent_{agent_idx}"
            precomputed[agent_id] = {
                "states": torch.stack(buffer.states[agent_id]).squeeze(1).to(self.device).detach(),
                "images": torch.stack(buffer.images[agent_id]).squeeze(1).to(self.device).detach(),
                "actions": torch.stack(buffer.actions[agent_id]).squeeze(1).to(self.device).detach(),
                "old_log_probs": torch.stack(buffer.log_probs[agent_id]).squeeze(-1).to(self.device).detach(),
                "advantages": advantages[agent_id].detach()
            }
        
        global_state_tensor = torch.stack(buffer.global_states).to(self.device).detach()

        losses_log = {'actor_losses': {f"agent_{i}": [] for i in range(self.num_agents)}, 'critic_losses': []}

        micro_batch_size = 256 
        num_micro_batches = len_global // micro_batch_size + (1 if len_global % micro_batch_size != 0 else 0)

        for epoch in range(5):
            for agent_idx in range(self.num_agents):
                agent_id = f"agent_{agent_idx}"
                actor = self.actors[agent_idx]
                data = precomputed[agent_id]
                
                self.actors_op[agent_idx].zero_grad()
                
                for start in range(0, len_global, micro_batch_size):
                    end = start + micro_batch_size
                    
                    states_b = data["states"][start:end]
                    images_b = data["images"][start:end]
                    actions_b = data["actions"][start:end]
                    old_probs_b = data["old_log_probs"][start:end]
                    adv_b = data["advantages"][start:end]

                    mean, std = actor(images_b, states_b)
                    dist = Normal(mean, std)
                    new_probs = dist.log_prob(actions_b).sum(dim=-1)
                    
                    ratio = torch.exp(new_probs - old_probs_b)
                    reinforce = ratio * adv_b
                    clipping = torch.clamp(ratio, 0.8, 1.2) * adv_b
                    
                    actor_loss = (-torch.min(reinforce, clipping).mean() - 0.05 * dist.entropy().mean()) / num_micro_batches
                    actor_loss.backward() 

                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                self.actors_op[agent_idx].step()
                losses_log['actor_losses'][agent_id].append(actor_loss.item() * num_micro_batches)

            self.critic_op.zero_grad()
            for start in range(0, len_global, micro_batch_size):
                end = start + micro_batch_size
                
                global_b = global_state_tensor[start:end]
                target_b = target_values[start:end]

                predicted_v = self.critic(global_b).squeeze(-1)
                critic_loss = (predicted_v - target_b).pow(2).mean() / num_micro_batches
                critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_op.step()
            losses_log['critic_losses'].append(critic_loss.item() * num_micro_batches)
        del precomputed
        torch.cuda.empty_cache()
        return losses_log


class BufferExp:
    def __init__(self):
        self.actions = {}
        self.log_probs = {}
        self.rewards = {}
        self.states = {}
        self.images = {}
        self.dones = {}
        self.critic_values = []
        self.global_states = []

    def store(self, actions_dict, log_probs_dict, rewards_dict, states_dict, images_dict,
              global_state, dones_dict, value):
        for agent_id in actions_dict.keys():
            if agent_id not in self.actions:
                self.actions[agent_id] = []
                self.log_probs[agent_id] = []
                self.rewards[agent_id] = []
                self.states[agent_id] = []
                self.images[agent_id] = []
                self.dones[agent_id] = []
        
        for agent_id in actions_dict.keys():
            self.actions[agent_id].append(actions_dict[agent_id].cpu())
            self.log_probs[agent_id].append(log_probs_dict[agent_id].cpu())
            self.rewards[agent_id].append(rewards_dict[agent_id])
            self.states[agent_id].append(states_dict[agent_id].cpu())
            self.images[agent_id].append(images_dict[agent_id].cpu())
            self.dones[agent_id].append(dones_dict[agent_id])
        
        self.global_states.append(global_state.cpu())
        self.critic_values.append(value.cpu())

    def clear_buffer(self):
        self.__init__()
        
    def __len__(self):
        return len(self.global_states)



