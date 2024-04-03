import os
import sys
import yaml
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.distributions import Categorical
from environment_generator import Environment
from models.mcts_vnet_model import MCTSVNet
from models.global_flow_model import GlobalFlowModel
from models.local_flow_model import LocalFlowTransformer
from utils.replay_buffer import ReplayBuffer
from utils.reward_normalizer import RewardNormalizer

# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class SimulationManager:
    def __init__(self, config_path, num_episodes, online_learning=False):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_episodes = num_episodes
        self.online_learning = online_learning
        self.env = Environment(config_path)
        self.global_flow_model, self.local_flow_model, self.mcts_vnet_model = self.initialize_models()
        self.replay_buffer = ReplayBuffer(self.config["training"]["max_buffer"])  # 初始化经验回放缓冲区，容量可配置
        self.reward_normalizer = RewardNormalizer()  # 初始化奖励标准化器
        if online_learning:
            self.optimizer = optim.Adam(self.mcts_vnet_model.parameters(), lr=self.config['training']["lr"])
            self.checkpoint_interval = self.config['training']["checkpoint_interval"]
            self.gamma = self.config['training']["gamma"]  # 折扣因子
            self.lambda_gae = self.config['training']["lambda_gae"]  # GAE参数
            self.entropy_coef = self.config['training']["entropy_coef"]  # 熵正则化系数
        else:
            self.optimizer = None
            self.checkpoint_interval = None
            self.gamma = None
            self.lambda_gae = None
            self.entropy_coef = None
        wandb.init(project="MCTSV-online_learning", config=self.config)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def initialize_models(self):
        global_flow_model = self.load_model(GlobalFlowModel(self.config['models']['global_flow_model']), self.config['datasets']['global_flow']['model_path'])
        local_flow_model = self.load_model(LocalFlowTransformer(self.config['models']['local_flow_model']), self.config['datasets']['local_flow']['model_path'])
        mcts_vnet_model = self.load_model(MCTSVNet(self.config['models']['mcts_vnet_model']), self.config['datasets']['mcts']['model_path'])
        return global_flow_model, local_flow_model, mcts_vnet_model

    def load_model(self, model_class, model_path):
        if os.path.exists(model_path):
            model_class.load_state_dict(torch.load(model_path, map_location=self.device))
        return model_class.to(self.device)

    def simulate_interaction(self):
        for episode in range(self.num_episodes):
            self.train_episode(episode)

    def train_episode(self, episode):
        self.state = self.env.reset()
        self.done = False
        episode_rewards = []

        while not self.done:
            global_matrix = torch.tensor(self.state['global_matrix'], dtype=torch.float).view(1, -1).to(self.device)
            local_matrix = torch.tensor(self.state['local_matrix'], dtype=torch.float).to(self.device)

            global_flow_output, local_flow_output = self.global_flow_model(global_matrix), self.local_flow_model(local_matrix)
            policy, _ = self.mcts_vnet_model(global_flow_output, local_flow_output)
            action_dist = Categorical(policy)
            action = action_dist.sample()

            next_state, reward, done = self.env.step(action=action.item(), use_mcts=self.config['training']["use_mcts"], use_mcts_to_train=self.config['training']["use_mcts_to_train"], global_flow_model=self.global_flow_model, local_flow_model=self.local_flow_model)
            
            # 下一状态（mctsv模型输入的处理）
            next_global_matrix = torch.tensor(next_state['global_matrix'], dtype=torch.float).view(1, -1).to(self.device)
            next_local_matrix = torch.tensor(next_state['local_matrix'], dtype=torch.float).to(self.device)

            next_global_flow_output = self.global_flow_model(next_global_matrix)
            next_local_flow_output = self.local_flow_model(next_local_matrix)

            normalized_reward = self.reward_normalizer.normalize(reward)
            episode_rewards.append(normalized_reward)
            
            self.replay_buffer.push((global_flow_output, local_flow_output), action.item(), normalized_reward, (next_global_flow_output, next_local_flow_output), done)

            self.state = next_state
            self.done = done

        # Sample experiences from the replay buffer to update the model
        if len(self.replay_buffer) >= self.config["training"]["BATCH_SIZE"]:
            experiences = self.replay_buffer.sample(self.config["training"]["BATCH_SIZE"])
            self.update_model(experiences, episode=episode)

        if (episode + 1) % self.checkpoint_interval == 0:
            self.save_model(episode, sum(episode_rewards))


    def update_model(self, experiences, episode):
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 解包states和next_states，因为它们是由两个不同大小的tensor组成的元组
        global_states, local_states = zip(*states)
        global_next_states, local_next_states = zip(*next_states)

        # 将每个states部分转换为合适的torch tensors并移至设备
        global_states = torch.stack(global_states).to(self.device)
        local_states = torch.stack(local_states).to(self.device)
        global_next_states = torch.stack(global_next_states).to(self.device)
        local_next_states = torch.stack(local_next_states).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # 对当前状态和下一个状态使用模型进行预测
        # 注意这里需要将global和local部分分别输入模型
        policy, current_values = self.mcts_vnet_model(global_states, local_states)
        next_policy, next_values = self.mcts_vnet_model(global_next_states, local_next_states)
        next_values = next_values.detach()

        # 假设policy的形状是[1, 128, 9]
        # 移除不必要的批次维度，使其变为[128, 9]
        policy = policy.squeeze(0)  # 移除第0维，因为它的大小为1
        next_policy = next_policy.squeeze(0)

        # GAE 计算
        returns, advantages = self.compute_gae(next_values, rewards, dones, current_values)

        # 计算损失函数
        action_probs = policy.gather(1, actions).squeeze()
        policy_loss = -(advantages.detach() * action_probs.log()).mean()
        value_loss = F.mse_loss(current_values.squeeze(), returns)

        # 更新模型
        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.mcts_vnet_model.parameters(), 0.5)
        self.optimizer.step()


        # 记录损失
        wandb.log({"loss": (policy_loss + value_loss).item(),
                    "value_loss": value_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "episode": episode,
                    "reward": sum(rewards),})

    def compute_gae(self, next_values, rewards, dones, values):
        gae = 0
        returns = []
        advantages = []

        # 张量形状变换
        next_values = next_values.squeeze(-1)  # 使用-1确保是移除最内侧的维度
        values = values.squeeze(-1)

        # 如果需要将这些值展平为一维张量，可以进一步使用 .view(-1)
        next_values = next_values.view(-1)
        values = values.view(-1)

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
        return torch.tensor(returns, dtype=torch.float, device=self.device), torch.tensor(advantages, dtype=torch.float, device=self.device)

    def save_model(self, episode, step_count, loss):
        model_save_path = os.path.join(self.config['datasets']['mcts']['save_model_path'], f"mcts_vnet_episode{episode}_step{step_count}_loss{loss}.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.mcts_vnet_model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    config_path = 'config/config.yml'
    num_episodes = 100  # Or fetch from config
    simulation_manager = SimulationManager(config_path, num_episodes, online_learning=True)
    simulation_manager.simulate_interaction()
