import os
import random
import sys
import numpy as np
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
from utils.replay_buffer import PrioritizedReplayBuffer, Experience
from utils.normalizer import Normalizer
from torch.optim import lr_scheduler

# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class SimulationManager:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_episodes = self.config["training"]["episodes"]
        self.online_learning = self.config["training"]["online_learning"]
        self.env = Environment(config_path)
        self.global_flow_model, self.local_flow_model, self.mcts_vnet_model = self.initialize_models()
        # 传入MCTSV_NET
        self.env.drone.mcts_vnet_model = self.mcts_vnet_model
        self.replay_buffer = PrioritizedReplayBuffer(self.config["training"]["max_buffer"])  # 初始化经验回放缓冲区，容量可配置
        self.normalizer = Normalizer()  # 初始化奖励标准化器
        self.loss_save = 0
        self.online_train_seed = self.config["environment"]["online_train_seed"]
        self.test_seed = self.config["environment"]["test_seed"]

        if self.online_learning:
            self.optimizer = optim.Adam(self.mcts_vnet_model.parameters(), lr=self.config['training']["lr"])
            # 初始化学习率调度器
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10)
            self.checkpoint_interval = self.config['training']["checkpoint_interval"]
            self.gamma = self.config['training']["gamma"]  # 折扣因子
            self.lambda_gae = self.config['training']["lambda_gae"]  # GAE参数
            self.entropy_coef = self.config['training']["entropy_coef"]  # 熵正则化系数
            self.clip_grad = self.config['training']["clip_grad"]  # 梯度裁剪阈值
            # 设置NumPy的随机种子
            np.random.seed(self.online_train_seed)
            # 设置Python标准库随机模块的种子
            random.seed(self.online_train_seed)
            torch.manual_seed(self.online_train_seed)
            torch.cuda.manual_seed_all(self.online_train_seed)  # 如果使用多个CUDA设备

        else:
            # 设置NumPy的随机种子
            np.random.seed(self.test_seed)
            # 设置Python标准库随机模块的种子
            random.seed(self.test_seed)
            torch.manual_seed(self.online_train_seed)
            torch.cuda.manual_seed_all(self.online_train_seed)  # 如果使用多个CUDA设备


        if self.config['training']['wandb']:
            wandb.init(project="MCTSV-online_learning", config=self.config)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def initialize_models(self):
        global_flow_model = self.load_model(GlobalFlowModel(self.config['models']['global_flow_model']), self.config['datasets']['global_flow']['model_path'])
        # local_flow_model = self.load_model(LocalFlowTransformer(self.config['models']['local_flow_model']), self.config['datasets']['local_flow']['model_path'])
        
        # global_flow_model = None
        local_flow_model = None
        mcts_vnet_model = self.load_model(MCTSVNet(self.config['models']['mcts_vnet_model']), self.config['datasets']['mcts']['model_path'])
        return global_flow_model, local_flow_model, mcts_vnet_model

    def load_model(self, model_class, model_path):
        if os.path.exists(model_path):
            model_class.load_state_dict(torch.load(model_path, map_location=self.device))
        return model_class.to(self.device)

    def simulate_interaction(self):
        for episode in range(self.num_episodes):
            if self.online_learning:
                self.mcts_vnet_model.train()
                self.train_episode(episode)

    def train_episode(self, episode):
        self.state = self.env.reset()
        self.done = False
        episode_rewards = []

        while not self.done:
            # self.env.run_simulation_animation()  # 运行画面模拟器（暂时存在一些问题）
            
            global_matrix = torch.tensor(self.state['global_matrix'], dtype=torch.float).to(self.device)
            local_matrix = torch.tensor(self.state['local_matrix'], dtype=torch.float).to(self.device)

            # global_flow可能会有潜在的问题
            global_flow_output = self.global_flow_model(global_matrix)
            
            # local_flow可能会有潜在的问题
            # local_flow_output = self.local_flow_model(local_matrix)
            
            # local_matrix在第0维上加一个维度
            local_matrix = local_matrix.unsqueeze(0)
            
            policy, _ = self.mcts_vnet_model(global_flow_output, local_matrix)
            action_dist = Categorical(policy)
            action = action_dist.sample()

            next_state, reward, done = self.env.step(action=action.item(), use_mcts=self.config['training']["use_mcts"], use_mcts_to_train=self.config['training']["use_mcts_to_train"], use_mcts_vnet_value=self.config['training']['use_mcts_vnet_value'], mcts_vnet_model=self.mcts_vnet_model, global_flow_model=self.global_flow_model, local_flow_model=self.local_flow_model)
            
            # use_mcts_to_train为True时，获取存在返回状态字典中的action，然后转换为index
            if self.config['training']["use_mcts"]:
                action = next_state['action']
                if action is None:
                    pass
                else:
                    action = self.env.drone.available_actions_dict_inv[action]
            
            # 下一状态（mctsv模型输入的处理）
            next_global_matrix = torch.tensor(next_state['global_matrix'], dtype=torch.float).to(self.device)
            next_local_matrix = torch.tensor(next_state['local_matrix'], dtype=torch.float).to(self.device)

            next_global_flow_output = self.global_flow_model(next_global_matrix)
            # next_local_flow_output = self.local_flow_model(next_local_matrix)

            # 奖励标准化
            normalized_reward = self.normalizer.normalize(reward)
            episode_rewards.append(normalized_reward)
            
            # 创建Experience对象
            experience = Experience((global_flow_output.detach().cpu().numpy(), local_matrix.detach().cpu().numpy()), action, normalized_reward, (next_global_flow_output.detach().cpu().numpy(), next_local_matrix.detach().cpu().numpy()), done)

            # 使用单个Experience对象作为参数调用push方法
            self.replay_buffer.push(experience)
            
            self.state = next_state
            self.done = done

        if self.config['training']['wandb']:
            # 记录奖励
            wandb.log({ "episode": episode,
                        "reward": sum(episode_rewards),
                        "collected_points": self.state['collected_points'],})

        # Sample experiences from the replay buffer to update the model
        if len(self.replay_buffer) >= self.config["training"]["BATCH_SIZE"]:
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.config["training"]["BATCH_SIZE"])
            self.update_model(states, actions, rewards, next_states, dones, weights, indices, episode=episode)


        if (episode + 1) % self.checkpoint_interval == 0:
            self.save_model(episode, sum(episode_rewards), self.loss_save)


    def update_model(self, states, actions, rewards, next_states, dones, weights, indices, episode):

        # 解包states和next_states，因为它们是由两个不同大小的tensor组成的元组
        global_states, local_states = states[0], states[1]
        global_next_states, local_next_states = next_states[0], next_states[1]
        
        global_states = global_states.to(self.device).detach()
        local_states = local_states.to(self.device).detach()
        global_next_states = global_next_states.to(self.device).detach()
        local_next_states = local_next_states.to(self.device).detach()

        # 分离计算图
        actions = actions.clone().detach().to(self.device).view(-1, 1).long()
        rewards = rewards.clone().detach().to(self.device).float()
        dones = dones.clone().detach().to(self.device).float()

        # 对当前状态和下一个状态使用模型进行预测
        # 注意这里需要将global和local部分分别输入模型
        policy, current_values = self.mcts_vnet_model(global_states, local_states)
        next_policy, next_values = self.mcts_vnet_model(global_next_states, local_next_states)

        # 分离计算图
        current_values = current_values.detach()
        next_values = next_values.detach()
        
        # 价值标准化
        current_values = self.normalizer.standardize(current_values)
        next_values = self.normalizer.standardize(next_values)
        
        policy = policy
        next_policy = next_policy

        # 增加重要性采样权重的应用
        weights = weights.clone().detach().to(self.device).float()

        # GAE 计算
        returns, advantages = self.compute_gae(next_values, rewards, dones, current_values)

        # 计算策略熵
        policy_entropy = -(policy * policy.log()).sum(1).mean()

        # 去除policy中的不必要的中间维度
        policy = policy.squeeze(1)  # 这将改变policy的形状为[batch_size, num_actions]

        # 计算损失函数
        action_probs = policy.gather(1, actions).squeeze()
        
        # 调整损失函数以考虑ISW
        policy_loss = -(advantages.detach() * action_probs.log() * weights).mean()
        value_loss = (F.mse_loss(current_values.squeeze(), returns) * weights).mean()
        
        # 在总损失中加入熵正则化项，注意是减去熵正则化项（因为我们想最大化熵）
        total_loss = policy_loss + value_loss - self.entropy_coef * policy_entropy

        # 更新模型
        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.mcts_vnet_model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.scheduler.step(sum(rewards))

        # 更新优先级
        with torch.no_grad():
            new_priorities = abs(current_values.squeeze() - returns).cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)


        self.loss_save = total_loss.item()

        if self.config['training']['wandb']:
            # 记录损失
            wandb.log({"loss": self.loss_save,
                        "value_loss": value_loss.item(),
                        "policy_loss": policy_loss.item(),})

    def compute_gae(self, next_values, rewards, dones, values):
        gae = 0
        returns = torch.zeros_like(rewards, device=self.device)
        advantages = torch.zeros_like(rewards, device=self.device)

        next_values = next_values.squeeze(-1).view(-1)
        values = values.squeeze(-1).view(-1)

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            returns[step] = gae + values[step]
            advantages[step] = gae
        
        return returns, advantages

    def save_model(self, episode, rewards, loss):
        model_save_path = os.path.join(self.config['datasets']['mcts']['save_model_path'], f"mcts_vnet_episode{episode+1}_avgRewards{rewards}_loss{loss}.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.mcts_vnet_model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    config_path = 'config/config.yml'
    simulation_manager = SimulationManager(config_path)
    simulation_manager.simulate_interaction()
