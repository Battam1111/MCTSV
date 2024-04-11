import copy
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


import torch

class Node:
    """
    Represents a node in the MCTS tree, optimized for both performance and clarity.
    This version includes action-value mapping for efficient storage and retrieval of action rewards.
    """
    
    def __init__(self, state, env, parent=None, action=None, depth=0):
        self.state = state  # 当前节点的状态
        self.env = env  # 环境对象，用于生成动作和进行模拟
        self.parent = parent  # 父节点
        self.action = action  # 导致该节点的动作
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.value = 0  # 节点的总价值
        self.depth = depth  # 节点的深度
        # 动作-价值映射，用于存储每个动作的平均奖励值
        self.action_values = defaultdict(float)
        # 为根节点初始化未尝试动作列表，子节点将从父节点继承
        self.untried_actions = env.drone._generate_available_actions()[0] if parent is None else parent.untried_actions[:]

    def add_child(self, state, action):
        """
        Adds a new child node for the given state and action.
        Removes the action from the list of untried actions.
        """
        # 从未尝试动作列表中移除该动作
        self.untried_actions.remove(action)
        new_node = Node(state, self.env, self, action, self.depth + 1)
        self.children.append(new_node)
        return new_node

    def update(self, reward):
        """
        Updates this node's statistics using a weighted average.
        The visit count is incremented, and the value is updated with the given reward.
        """
        self.visits += 1
        alpha = 1 / self.visits  # Weight for the new reward
        self.value = (1 - alpha) * self.value + alpha * reward
        if self.action is not None:
            self.action_values[self.action] = (1 - alpha) * self.action_values[self.action] + alpha * reward

    def is_fully_expanded(self):
        """
        Checks if this node has expanded all possible actions.
        """
        return len(self.untried_actions) == 0
    
    def ucb_score(self, c_param=1.4):
        """
        Computes the Upper Confidence Bound (UCB) score for this node.
        """
        if self.visits == 0:
            return float('inf')  # 避免除以零
        return (self.value / self.visits) + c_param * np.sqrt((2 * np.log(self.parent.visits) / self.visits))

class MCTS:
    """
    An optimized version of the Monte Carlo Tree Search (MCTS) algorithm.
    This implementation focuses on efficiency, readability, and flexibility.
    """
    
    def __init__(self, environment, num_simulations, depth_threshold=10):
        self.environment = environment
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_threshold = depth_threshold  # 深度阈值
        
    def adaptive_num_simulations(self, node):
        """
        Adjusts the number of simulations dynamically based on the depth of the node,
        to balance exploration and exploitation across different stages of the search.
        """
        depth = self._get_depth(node)
        base_num_simulations = self.num_simulations
        return max(1, base_num_simulations - depth)
    
    def _get_depth(self, node):
        """
        Computes the depth of the given node in the tree.
        """
        depth = 0
        while node.parent is not None:
            node = node.parent
            depth += 1
        return depth
    
    def best_child_bayesian_optimization(self, node, c_param=1.4):
        """
        Selects the best child using Bayesian Optimization.
        """
        def objective(x):
            # 转换 x 为离散动作索引
            action_index = int(x[0])
            if action_index >= len(node.children):
                return float('inf')  # 超出范围的动作索引将被排除
            return -node.children[action_index].ucb_score(c_param)

        # 初始化高斯过程回归模型
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel)

        # 收集子节点的动作索引和对应的 UCB 得分
        x_obs = np.array([[i] for i in range(len(node.children))])
        y_obs = np.array([[-child.ucb_score(c_param)] for child in node.children])

        # 更新高斯过程回归模型
        gp.fit(x_obs, y_obs)

        # 使用贝叶斯优化找到最佳动作索引
        res = minimize(lambda x: -gp.predict(x.reshape(1, -1), return_std=True)[0], x0=[0], bounds=[(0, len(node.children) - 1)], method='L-BFGS-B')

        best_action_index = int(res.x[0])
        return node.children[best_action_index]
    
    def _simulate(self, node, global_flow_model, local_flow_model):
        """
        Simulates a play-through from the given node using the environment's dynamics
        and the provided flow models, returning the total accumulated reward.
        """
        simulation_env = copy.deepcopy(self.environment)
        simulation_env.is_simulated = True
        simulation_env.set_state(node.state)
        total_reward = 0
        
        while not simulation_env._check_done():
            global_matrix = torch.tensor(simulation_env.state['global_matrix'], dtype=torch.float).to(self.device)
            local_matrix = torch.tensor(simulation_env.state['local_matrix'], dtype=torch.float).to(self.device)
            # local_matrix在第0维上加一个维度
            local_matrix = local_matrix.unsqueeze(0)
            
            global_flow_output = global_flow_model(global_matrix)
            # local_flow_output = local_flow_model(local_matrix)
            action = simulation_env.drone.sample_action(global_flow_output, local_matrix, is_greed=True)
            
            _, reward, _ = simulation_env.step(action=action, global_flow_model=global_flow_model, local_flow_model=local_flow_model, is_simulated=True)
            total_reward += reward
        
        return total_reward
    
    def _search(self, state, global_flow_model, local_flow_model):
        """
        Performs the MCTS search from the given state, using the specified models.
        """
        root_node = Node(state, self.environment)
        num_simulations = self.adaptive_num_simulations(root_node)
        
        for _ in range(num_simulations):
            node = root_node
            
            if node.depth >= self.depth_threshold:
                break  # 如果节点深度超过阈值，则停止扩展
            
            # Perform Selection
            while node.is_fully_expanded() and not self.environment._check_done(node.state):
                node = self.best_child_bayesian_optimization(node)  # 使用贝叶斯优化选择最佳子节点
            
            # Perform Expansion
            if not node.is_fully_expanded():
                action = node.untried_actions[0]
                new_state, reward, done = self.environment.step(global_flow_model=global_flow_model, local_flow_model=local_flow_model, action=action, is_simulated=True)
                if not done:
                    node = node.add_child(new_state, action)
            
            # Perform Simulation
            reward = self._simulate(node, global_flow_model, local_flow_model)
            
            # Perform Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent
        
        return root_node.best_child().action if root_node.best_child() else None
    
    def search(self, initial_state, global_flow_model, local_flow_model):
        """
        Public method to initiate the MCTS search given the initial state and models.
        Returns the action selected by the MCTS algorithm.
        """
        return self._search(initial_state, global_flow_model, local_flow_model)