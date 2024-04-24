import copy
import os
import random
import sys
import numpy as np
import yaml
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical
from environment_generator import Environment
from models.mctsv_1d import MCTSVNet
from models.global_flow_model import GlobalFlowModel
from models.local_flow_model import LocalFlowTransformer
from src.mcts_algorithm import MCTS
from utils.replay_buffer import PrioritizedReplayBuffer, Experience
from utils.normalizer import Normalizer
from torch.optim import lr_scheduler

# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def generate_run_name(config):
    parts = ["MCTSV"]
    if config["training"]["online_learning"]:
        parts.append("Train")
    else:
        parts.append("Test")
    if config["training"]["use_mcts"]:
        parts.append("withMCTS")
        parts.append("{0}Simulations".format(config["mcts"]["num_simulations"]))
        if config["training"]["use_mcts_vnet_value"]:
            parts.append("withVNetValue")
    parts.append("memoryBATCH{0}".format(config["training"]["BATCH_SIZE"]))
    parts.append("{0}Episodes".format(config["training"]["episodes"]))
    return "-".join(parts)

class SimulationManager:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.run_name = generate_run_name(self.config)

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
            if self.config['training']['wandb']:
                wandb.init(project="MCTSV-online_learning", name=self.run_name, config=self.config)

        else:
            # 测试时在线学习
            self.optimizer = optim.Adam(self.mcts_vnet_model.parameters(), lr=self.config['training']["lr"])
            # 初始化学习率调度器
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10)
            self.checkpoint_interval = self.config['training']["checkpoint_interval"]
            self.gamma = self.config['training']["gamma"]  # 折扣因子
            self.lambda_gae = self.config['training']["lambda_gae"]  # GAE参数
            self.entropy_coef = self.config['training']["entropy_coef"]  # 熵正则化系数
            self.clip_grad = self.config['training']["clip_grad"]  # 梯度裁剪阈值

            # 设置NumPy的随机种子
            np.random.seed(self.test_seed)
            # 设置Python标准库随机模块的种子
            random.seed(self.test_seed)
            torch.manual_seed(self.test_seed)
            torch.cuda.manual_seed_all(self.test_seed)  # 如果使用多个CUDA设备
            if self.config['testing']['wandb']:
                wandb.init(project="MCTSV-online_learning", name=self.run_name, config=self.config)


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
        if not self.config['training']['online_learning']:
            if os.path.exists(model_path):
                try:
                    # 尝试加载模型权重
                    model_state_dict = torch.load(model_path, map_location=self.device)
                    model_class.load_state_dict(model_state_dict)
                except RuntimeError as e:
                    # 如果发生错误，打印错误并且检查键的不匹配
                    print(f"Error loading model weights from {model_path}: {e}")
                    print("\nModel's expected state_dict:")
                    for param_tensor in model_class.state_dict():
                        print(param_tensor, "\t", model_class.state_dict()[param_tensor].size())
                    print("\nLoaded state_dict keys:")
                    for k in model_state_dict:
                        print(k)
                    # 可以选择在这里使用 strict=False 以继续加载程序，但注意这可能隐藏问题
                    model_class.load_state_dict(model_state_dict, strict=False)
            else:
                print(f"Model path {model_path} does not exist.")
        return model_class.to(self.device)

    def simulate_interaction(self):
        if self.online_learning:
            num_episodes = self.config["training"]["episodes"]
            self.mcts_vnet_model.train()
            for episode in range(num_episodes):
                self.train_episode(episode)
        else:
            num_episodes = self.config["testing"]["episodes"]
            self.mcts_vnet_model.eval()
            for episode in range(num_episodes):
                self.test_episode(episode)

    def test_episode(self, episode):
        self.state = self.env.reset()
        self.done = False
        episode_rewards = []

        while not self.done:
            
            global_matrix = torch.tensor(self.state['global_matrix'], dtype=torch.float).to(self.device)
            local_matrix = torch.tensor(self.state['local_matrix'], dtype=torch.float).to(self.device)

            # global_flow可能会有潜在的问题
            global_flow_output = self.global_flow_model(global_matrix)
            
            # local_flow可能会有潜在的问题
            # local_flow_output = self.local_flow_model(local_matrix)
            
            # local_matrix在第0维上加一个维度，因为未经过局部流模型
            local_matrix = local_matrix.unsqueeze(0)

            if self.config['testing']["use_mcts"]:
                mcts = MCTS(copy.deepcopy(self.env), num_simulations=self.config['mcts']['num_simulations'])
                # 使用 MCTS 算法来获取最佳动作
                action = mcts.search(copy.deepcopy(self.state), self.mcts_vnet_model, self.global_flow_model, self.local_flow_model, self.config['testing']["use_mcts_vnet_value"])
                action = self.env.drone.available_actions_dict_inv[action]
                # 测试时在线学习用到
                policy, _ = self.mcts_vnet_model(global_flow_output, local_matrix)
            elif self.config['testing']['random']:
                action = random.choice(self.env.drone.available_actions)
            else:
                policy, _ = self.mcts_vnet_model(global_flow_output, local_matrix)
                action_dist = Categorical(policy)
                action = action_dist.sample().item()
                
            next_state, reward, done = self.env.step(action=action, use_mcts_to_train=self.config['testing']["use_mcts_to_train"], global_flow_model=self.global_flow_model, local_flow_model=self.local_flow_model)
            
            # 测试时在线学习
            if self.config['testing']['online_learning']:
                # 下一状态（mctsv模型输入的处理）
                next_global_matrix = torch.tensor(next_state['global_matrix'], dtype=torch.float).to(self.device)
                next_local_matrix = torch.tensor(next_state['local_matrix'], dtype=torch.float).to(self.device)

                next_global_flow_output = self.global_flow_model(next_global_matrix)
                # next_local_flow_output = self.local_flow_model(next_local_matrix)

                # 奖励标准化
                normalized_reward = self.normalizer.normalize(reward)
                episode_rewards.append(reward)
                
                # 创建Experience对象
                experience = Experience((global_flow_output.detach().cpu().numpy(), local_matrix.detach().cpu().numpy()), action, normalized_reward, (next_global_flow_output.detach().cpu().numpy(), next_local_matrix.detach().cpu().numpy()), done)

                # 使用单个Experience对象作为参数调用push方法
                self.replay_buffer.push(experience)


            self.state = next_state
            self.done = done
            episode_rewards.append(normalized_reward)
            # 应该用这个，但可以之后重新跑一次
            # episode_rewards.append(reward)

        # 测试时在线学习
        if self.config['testing']['online_learning']:
            # Sample experiences from the replay buffer to update the model
            if len(self.replay_buffer) >= self.config["training"]["BATCH_SIZE"]:
                states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.config["training"]["BATCH_SIZE"])
                self.update_model(states, actions, rewards, next_states, dones, weights, indices)

        if self.config['environment']['is_animation']:
            self.env.visualizer.save_animation(filename=f'saved_animations/{self.run_name}/animation', episode=episode)

        if self.config['testing']['wandb']:
            # 记录奖励
            wandb.log({ "episode": episode,
                        "reward": sum(episode_rewards),
                        "collected_points": self.state['collected_points'],
                        "collisions": self.state['collisions'],
                        "max_num_signal_points": self.env.max_num_signal_points,
                        "num_obstacles": self.env.num_obstacles})

    def train_episode(self, episode):
        self.state = self.env.reset()
        self.done = False
        episode_rewards = []

        while not self.done:
            
            global_matrix = torch.tensor(self.state['global_matrix'], dtype=torch.float).to(self.device)
            local_matrix = torch.tensor(self.state['local_matrix'], dtype=torch.float).to(self.device)

            # global_flow可能会有潜在的问题
            global_flow_output = self.global_flow_model(global_matrix)
            
            # local_flow可能会有潜在的问题
            # local_flow_output = self.local_flow_model(local_matrix)
            
            # local_matrix在第0维上加一个维度
            local_matrix = local_matrix.unsqueeze(0)
            
            if self.config['training']["use_mcts"]:
                mcts = MCTS(copy.deepcopy(self.env), num_simulations=self.config['mcts']['num_simulations'])
                # 使用 MCTS 算法来获取最佳动作
                action = mcts.search(copy.deepcopy(self.state), self.mcts_vnet_model, self.global_flow_model, self.local_flow_model, self.config['training']["use_mcts_vnet_value"])
                del mcts
            else:
                policy, _ = self.mcts_vnet_model(global_flow_output, local_matrix)
                action_dist = Categorical(policy)
                action = action_dist.sample().item()

            next_state, reward, done = self.env.step(action=action, use_mcts_to_train=self.config['training']["use_mcts_to_train"], global_flow_model=self.global_flow_model, local_flow_model=self.local_flow_model)
            
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
                        "collected_points": self.state['collected_points'],
                        "collisions": self.state['collisions']})

        # Sample experiences from the replay buffer to update the model
        if len(self.replay_buffer) >= self.config["training"]["BATCH_SIZE"]:
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.config["training"]["BATCH_SIZE"])
            self.update_model(states, actions, rewards, next_states, dones, weights, indices)

        if (episode + 1) % self.checkpoint_interval == 0:
            self.save_model(episode, sum(episode_rewards), self.loss_save)
            if self.config['environment']['is_animation']:
                self.env.visualizer.save_animation(filename=f'saved_animations/{self.run_name}/animation', episode=episode)

    def update_model(self, states, actions, rewards, next_states, dones, weights, indices):

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
        value_loss = F.huber_loss(current_values.squeeze(), returns, delta=1.0) * weights.mean()
        
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
        # 格式化rewards和loss以改善文件名的可读性
        formatted_rewards = f"{rewards:.2f}"
        formatted_loss = f"{loss:.2f}"
        
        # 去除不必要的开头斜杠，确保路径是相对的
        model_save_path = os.path.join(self.config['datasets']['mcts']['save_model_path'], self.run_name, f"episode{episode+1}_avgRewards{formatted_rewards}_loss{formatted_loss}.pt")
        
        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        try:
            # 尝试保存模型状态
            torch.save(self.mcts_vnet_model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

if __name__ == "__main__":
    config_path = 'config/config.yml'
    simulation_manager = SimulationManager(config_path)
    simulation_manager.simulate_interaction()
