import numpy as np
import torch
import copy
from collections import defaultdict
from torch.distributions import Categorical

class Node:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state, env, parent=None, action=None):
        self.state = state
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        # 为根节点初始化未尝试动作列表，子节点将从父节点继承
        self.untried_actions = env.drone._generate_available_actions()[0] if parent is None else parent.untried_actions[:]

    def add_child(self, state, action):
        # 创建子节点时从未尝试动作列表中移除该动作
        self.untried_actions.remove(action)
        new_node = Node(state, self.env, self, action)
        self.children.append(new_node)
        return new_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        """
        Returns the best child node based on the UCT score.

        Args:
        - c_param (float): Exploration parameter.

        Returns:
        - Node: The best child node according to the UCT score.
        """
        if not self.children:
            return None  # 当前节点没有子节点，返回None

        # 计算每个子节点的UCT分数
        choices_weights = [
            (child.value / (child.visits + 1)) + c_param * np.sqrt((2 * np.log(self.visits + 1) / (child.visits + 1)))
            for child in self.children
        ]

        # 选择并返回UCT分数最高的子节点
        best_index = np.argmax(choices_weights)
        return self.children[best_index]



class MCTS:
    """
    Monte Carlo Tree Search algorithm implementation.
    """
    def __init__(self, environment, num_simulations):
        self.environment = environment
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def adaptive_num_simulations(self, node):
        # 自适应调整模拟次数的逻辑
        depth = self.get_depth(node)
        base_num_simulations = self.num_simulations  # 基础模拟次数
        return max(1, base_num_simulations - depth)

    def get_depth(self, node):
        depth = 0
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth

    def simulate(self, node, global_flow_model, local_flow_model):
        current_env = copy.deepcopy(self.environment)
        current_env.is_simulated = True
        current_env.set_state(node.state)
        total_reward = 0

        while not current_env._check_done():
            global_matrix = torch.tensor(current_env.state['global_matrix'], dtype=torch.float).view(1, -1).to(self.device)
            local_matrix = torch.tensor(current_env.state['local_matrix'], dtype=torch.float).to(self.device)

            global_flow_output = global_flow_model(global_matrix)
            local_flow_output = local_flow_model(local_matrix)

            action = current_env.drone.sample_action(global_flow_output, local_flow_output, is_greed=True)
            # action = Categorical(action_probs).sample().item()

            _, reward, _ = current_env.step(action=action, global_flow_model=global_flow_model, local_flow_model=local_flow_model, is_simulated=True)
            total_reward += reward

        return total_reward

    def search(self, state, global_flow_model, local_flow_model):
        root_node = Node(state, self.environment)

        num_simulations = self.adaptive_num_simulations(root_node)

        for _ in range(num_simulations):
            node = root_node
            simulation_env = copy.deepcopy(self.environment)
            simulation_env.is_simulated = True
            simulation_env.set_state(node.state)

            # Selection
            while not simulation_env._check_done():
                if node.is_fully_expanded():
                    next_node = node.best_child()
                    if next_node is None:
                        break  # 如果没有更多的子节点可供选择，跳出循环
                    node = next_node  # 移至选择的最佳子节点
                    simulation_env.set_state(node.state)
                else:
                    break  # 找到未完全扩展的节点，准备扩展
                    
            # Expansion
            if not node.is_fully_expanded() and not simulation_env._check_done():
                action = node.untried_actions[0]  # 选择第一个未尝试的动作进行扩展
                new_state, reward, done = simulation_env.step(action=action, global_flow_model=global_flow_model, local_flow_model=local_flow_model, is_simulated=True)
                if not done:
                    node.add_child(new_state, action)  # 添加新的子节点，但不立即跳转到这个新节点

            # Simulation
            reward = self.simulate(node, global_flow_model, local_flow_model)

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        return root_node.best_child().action if root_node.best_child() else None

