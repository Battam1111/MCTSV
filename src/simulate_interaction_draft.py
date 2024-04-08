
import yaml
import torch
import os
import sys
import torch.optim as optim
import torch.nn.functional as F
import wandb

# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment_generator import Environment
from models.mcts_vnet_model import MCTSVNet
from models.global_flow_model import GlobalFlowModel
from models.local_flow_model import LocalFlowTransformer
from torch.distributions import Categorical


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulate_interaction(config_path, num_episodes, online_learning=False):
    
    # 加载配置文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 初始化wandb
    wandb.init(project="MCTSV-online_learning", config=config)

    # 初始化环境
    env = Environment(config_path)

    # 初始化全局流模型和局部流模型
    global_flow_model_path = config['datasets']['global_flow']['model_path']
    local_flow_model_path = config['datasets']['local_flow']['model_path']

    if global_flow_model_path and os.path.exists(global_flow_model_path):
        global_flow_model = GlobalFlowModel(config['models']['global_flow_model'])
        global_flow_model.load_state_dict(torch.load(global_flow_model_path, map_location=device))
    else:
        global_flow_model = GlobalFlowModel(config['models']['global_flow_model'])

    if local_flow_model_path and os.path.exists(local_flow_model_path):
        local_flow_model = LocalFlowTransformer(config['models']['local_flow_model'])
        local_flow_model.load_state_dict(torch.load(local_flow_model_path, map_location=device))
    else:
        local_flow_model = LocalFlowTransformer(config['models']['local_flow_model'])

    # 初始化 MCTSVNet
    mcts_vnet_model_path = config['datasets']['mcts']['model_path']
    if mcts_vnet_model_path and os.path.exists(mcts_vnet_model_path):
        mcts_vnet_model = MCTSVNet(config['models']['mcts_vnet_model'])
        mcts_vnet_model.load_state_dict(torch.load(mcts_vnet_model_path, map_location=device))
    else:
        mcts_vnet_model = MCTSVNet(config['models']['mcts_vnet_model'])


    # 将 MCTSVNet 模型传入无人机类
    env.drone.mcts_vnet_model = mcts_vnet_model

    if online_learning:
        optimizer = optim.Adam(mcts_vnet_model.parameters(), lr=config['training']["lr"])
        checkpoint_interval = config['training']["checkpoint_interval"]
        gamma = config['training']["gamma"]  # 折扣因子
        lambda_gae = config['training']["lambda_gae"]  # GAE参数
        entropy_coef = config['training']["entropy_coef"]  # 熵正则化系数

    # 模拟交互
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        rewards, values, log_probs, entropies = [], [], [], []

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # env.run_simulation_animation()  # 运行画面模拟器（暂时存在一些问题）

            global_matrix = torch.tensor(state['global_matrix'], dtype=torch.float).view(1, -1).to(device)
            local_matrix = torch.tensor(state['local_matrix'], dtype=torch.float).to(device)

            # global_flow_output = global_flow_model(global_matrix)
            # local_flow_output = local_flow_model(local_matrix)

            # 获取策略和价值
            mcts_vnet_model.train()

            # policy, value = mcts_vnet_model(global_matrix, local_matrix)

            # 使用两个transformer流模型的输出作为输入
            policy, value = mcts_vnet_model(global_matrix, local_matrix)
            action_dist = Categorical(policy)

            # 在线学习抽样动作训练
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            # 执行动作
            next_state, reward, done = env.step(action=action, use_mcts=config['training']["use_mcts"], use_mcts_to_train=config['training']["use_mcts_to_train"], global_flow_model=global_flow_model, local_flow_model=local_flow_model)  # 执行动作并获取下一个状态和奖励
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state  # 更新状态

            # 处理序列末端
        next_value = torch.tensor([0.0], device=device) if done else mcts_vnet_model(global_matrix, local_matrix)[1].detach()
        values.append(next_value)

        # 计算GAE和优化模型
        gae, returns = 0, []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * lambda_gae * gae
            returns.insert(0, gae + values[step])

        policy_loss, value_loss, gae_loss = 0, 0, 0
        for log_prob, value, ret in zip(log_probs, values, returns):
            advantage = ret - value.item()
            policy_loss += -log_prob * advantage - entropy_coef * entropies[step]
            value_loss += F.smooth_l1_loss(value, torch.tensor([[ret]]).to(device))

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mcts_vnet_model.parameters(), 0.5)
        optimizer.step()

        # 记录损失
        wandb.log({"loss": loss.item(),
                    "value_loss": value_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "episode": episode, "step": step_count,
                    "reward": sum(rewards),})

        mcts_vnet_model.eval()  # 将模型设置为评估模式

        if step_count % checkpoint_interval == 0:
            if not os.path.exists(config['datasets']['mcts']['save_model_path']):
                os.makedirs(config['datasets']['mcts']['save_model_path'])

            model_path = f"mcts_vnet_episode{episode}_step{step_count}_loss{loss.item()}.pt"
            torch.save(mcts_vnet_model.state_dict(), os.path.join(config['datasets']['mcts']['save_model_path'], model_path))
            print(f"Model saved to {model_path}")

        step_count += 1

if __name__ == "__main__":
    config_path = 'config/config.yml'
    num_episodes = 100  # 模拟的交互次数
    simulate_interaction(config_path, num_episodes, online_learning=True)
