import torch
from torch.distributions import Normal
from torch.optim import Adam
from models.networks import Actor_network, Critic_Actor


class MAPPO:
    """Algoritmo MAPPO.
    Definimos como aprender y actualizar la política de los actores y el crítico"""
    def __init__(self, num_agents, space_obs, space_act, gamma, par_lambda, device):
        """Constructor: donde vamos a ejecutar, creacion de redes, definición del optimizador"""
        self.device = device
        self.num_agents = num_agents
        self.agent_id_to_idx = {f"agent_{i}": i for i in range(num_agents)}
     
        self.actors = [Actor_network(space_obs, space_act).to(self.device) for _ in range(num_agents)]
        self.critic = Critic_Actor(space_obs * num_agents).to(self.device)

        self.actors_op = [Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_op = Adam(self.critic.parameters(), lr=3e-4)
        self.gamma = gamma
        self.par_lambda = par_lambda


    def politic(self, state, cam_state, agent_id):
        """Politica de los agentes. Seguirán una distribución Gaussiana"""
        if isinstance(agent_id, str):
            actor_id = self.agent_id_to_idx[agent_id]
        else:
            actor_id = agent_id

        actor = self.actors[actor_id]
        mean, std = actor(state, cam_state)

        #a veces llega un NaN y se carga el entrenamiento, por prevenir
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            print(f"[WARN] NaN/Inf en la politica de {agent_id}, usando accion neutra")
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = torch.nan_to_num(std, nan=0.1, posinf=0.1, neginf=0.1).clamp(min=1e-3)

        dist = Normal(mean, std)
        action_to_buffer = dist.sample()
        
        throttle = torch.tanh(action_to_buffer[:, 0:1])
        steer = torch.tanh(action_to_buffer[:, 1:2])
        action_tensor = torch.cat([throttle, steer], dim=1)

        prob = dist.log_prob(action_to_buffer).sum(dim=-1)
        

        return action_tensor, prob, action_to_buffer
    
    def critic_evaluation(self, state_final, state_cam):
        """Evaluación del crítico"""
        return self.critic(state_final, state_cam)
    

    def update(self, buffer):
        """Actualizacion de la politica que aprenden los agentes y el crítico.
        A este último se le hace enfasis para que tenga mas casos de prueba"""
        advantages = {}
        targets = {}
        values = torch.stack(buffer.critic_values).squeeze(-1).to(self.device).detach()
        len_global = len(buffer.global_states)

        for agent_id in buffer.rewards.keys():
            agent_adv = []
            gae = 0
            for t in reversed(range(len_global)):
                done_t = bool(buffer.dones[agent_id][t])
                mask = 0.0 if done_t else 1.0
                next_val = 0.0 if (t == len_global - 1 or done_t) else values[t+1]
                reward = buffer.rewards[agent_id][t]
                delta = reward + self.gamma * next_val - values[t]
                gae = delta + self.gamma * self.par_lambda * mask * gae
                agent_adv.insert(0, gae)

            advi = torch.stack(agent_adv).to(self.device)
            targets[agent_id] = advi + values
            advantages[agent_id] = (advi - advi.mean()) / (advi.std() + 1e-7)

        target_values = torch.stack([targets[aid] for aid in targets]).mean(dim=0).detach()

        precomputed = {}
        for agent_idx in range(self.num_agents):
            agent_id = f"agent_{agent_idx}"
            precomputed[agent_id] = {
                "states": torch.stack(buffer.states[agent_id]).squeeze(1).to(self.device).detach(),
                "cam_states": torch.stack(buffer.cam_states[agent_id]).squeeze(1).to(self.device).detach(),
                "actions": torch.stack(buffer.actions[agent_id]).squeeze(1).to(self.device).detach(),
                "old_log_probs": torch.stack(buffer.log_probs[agent_id]).squeeze(-1).to(self.device).detach(),
                "advantages": advantages[agent_id].detach()
            }
        
        global_state_tensor = torch.stack(buffer.global_states).squeeze(1).to(self.device).detach()
        global_cam_tensor = torch.stack(buffer.global_states_cam).squeeze(1).to(self.device).detach()

        losses_log = {'actor_losses': {f"agent_{i}": [] for i in range(self.num_agents)}, 'critic_losses': []}


        mb_size = 256
        #5 epoch para actualizar la politica de los agentes viendo si es mejor o peor de lo que teniamos
        for epoch in range(5):
            for agent_idx in range(self.num_agents):
                agent_id = f"agent_{agent_idx}"
                actor = self.actors[agent_idx]
                data = precomputed[agent_id]

                mean, std = actor(data["states"], data["cam_states"])
                dist = Normal(mean, std)
                new_probs = dist.log_prob(data["actions"]).sum(dim=-1)

                #clamp del exponente para evitar reventar el update
                ratio = torch.exp((new_probs - data["old_log_probs"]).clamp(-10.0, 10.0))
                reinforce = ratio * data["advantages"]
                clipping = torch.clamp(ratio, 0.8, 1.2) * data["advantages"]
                actor_loss = -torch.min(reinforce, clipping).mean() - 0.01 * dist.entropy().mean()

                self.actors_op[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                self.actors_op[agent_idx].step()
                losses_log['actor_losses'][agent_id].append(actor_loss.item())

            #mas updates buscando un mejor critico y que a la larga se tenga mejor conocimiento del entorno
            perm = torch.randperm(len_global, device=self.device)
            for start in range(0, len_global, mb_size):
                mb = perm[start:start + mb_size]
                predicted_v = self.critic(global_state_tensor[mb], global_cam_tensor[mb]).squeeze(-1)
                critic_loss = (predicted_v - target_values[mb]).pow(2).mean()

                self.critic_op.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_op.step()
                losses_log['critic_losses'].append(critic_loss.item())

        del precomputed
        torch.cuda.empty_cache()
        return losses_log


class BufferExp:
    """Buffer de experiencias"""
    def __init__(self):
        """Constructor para cada diccionario que necesitemos"""
        self.actions = {}
        self.log_probs = {}
        self.rewards = {}
        self.states = {}
        self.cam_states ={}
        self.dones = {}
        self.critic_values = []
        self.global_states = []
        self.global_states_cam = []

    def store(self, actions_dict, log_probs_dict, rewards_dict, states_dict, cam_dict,
              global_state, global_state_cam, dones_dict, value):
        """Guardamos los datos en nuestros diccionarios"""
        for agent_id in actions_dict.keys():
            if agent_id not in self.actions:
                self.actions[agent_id] = []
                self.log_probs[agent_id] = []
                self.rewards[agent_id] = []
                self.states[agent_id] = []
                self.cam_states[agent_id] = []
                self.dones[agent_id] = []
        
        for agent_id in actions_dict.keys():
            self.actions[agent_id].append(actions_dict[agent_id].cpu())
            self.log_probs[agent_id].append(log_probs_dict[agent_id].cpu())
            self.rewards[agent_id].append(rewards_dict[agent_id])
            self.states[agent_id].append(states_dict[agent_id].cpu())
            self.cam_states[agent_id].append(cam_dict[agent_id].cpu())
            self.dones[agent_id].append(dones_dict[agent_id])
        
        self.global_states.append(global_state.cpu())
        self.global_states_cam.append(global_state_cam.cpu())
        self.critic_values.append(value.cpu())

    def clear_buffer(self):
        """Limpiamos el buffer para no acumular basura entre episodios"""
        self.__init__()
        
    def __len__(self):
        """Por si necesitamos obtener la longitud del buffer"""
        return len(self.global_states)



