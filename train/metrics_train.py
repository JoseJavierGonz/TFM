import json
import numpy as np
import matplotlib.pyplot as plt

class TrainingMetrics:
    def __init__(self, save_path='training_metrics.json'):
        self.save_path = save_path
        self.data = {
            'episodes': [],
            'agent_0': [],
            'agent_1': [],
            'actor_loss_avg': [],
            'critic_loss': [],
            'reward_avg': []
        }
    
    def log_episode(self, episode, actor_loss_dict, critic_loss_list, reward_avg):
 
        self.data['episodes'].append(episode)
        
        agent_0_loss = np.mean(actor_loss_dict['agent_0']) if 'agent_0' in actor_loss_dict else 0
        agent_1_loss = np.mean(actor_loss_dict['agent_1']) if 'agent_1' in actor_loss_dict else 0
        
        self.data['agent_0'].append(agent_0_loss)
        self.data['agent_1'].append(agent_1_loss)
        self.data['actor_loss_avg'].append((agent_0_loss + agent_1_loss) / 2)
        
        critic_avg = np.mean(critic_loss_list) if critic_loss_list else 0
        self.data['critic_loss'].append(critic_avg)
        
        self.data['reward_avg'].append(reward_avg)
        
        self.save()
    
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
