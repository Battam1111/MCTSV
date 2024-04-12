import copy
import numpy as np
from collections import defaultdict
import time
import torch


class Node:
    def __init__(self, state, env, parent=None, action=None, depth=0, c_param=1.4):
        self.state = state
        self.env = env
        self.parent = parent  
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.depth = depth
        self.action_values = defaultdict(float)
        self.untried_actions = env.drone._generate_available_actions()[0] if parent is None else parent.untried_actions[:]
        self.c_param = c_param
        self.ucb_score = float('inf')
        self.total_log_parent_visits = np.log(1)  # 初始化为对1取对数

    def update_ucb_log_terms(self):
        if self.parent:
            self.total_log_parent_visits = np.log(self.parent.visits + 1)  # 父节点访问次数更新时调用

    def calculate_ucb_score(self, child):
        """
        使用缓存的对数值计算UCB分数
        """
        exploitation = child.value / (child.visits + 1e-5)
        exploration = self.c_param * np.sqrt(self.total_log_parent_visits / (child.visits + 1e-5))
        return exploitation + exploration

    def dispose(self):
        # Explicitly disposes of node resources and references to assist with garbage collection
        self.state = None
        self.env = None
        if self.children:
            for child in self.children.values():
                child.dispose()  # Ensure child nodes are also disposed
        self.parent = None
        self.action = None
        self.children.clear()
        self.action_values.clear()
        self.untried_actions = None

    def importance(self):
        """
        计算节点的重要性。这里简单地使用访问次数作为重要性的度量。
        """
        return self.visits

    def uncertainty(self):
        """
        计算节点的不确定性。使用子节点价值的标准差作为不确定性的度量。
        """
        if not self.children:
            return 0
        child_values = np.array([child.value for child in self.children])
        return np.std(child_values)

    def add_child(self, state, action):
        """
        Adds a new child node for the given state and action.
        Updates the list of untried actions dynamically.
        """
        new_node = Node(state, self.env, self, action, self.depth + 1)
        self.children[action] = new_node  # 使用动作作为键存储子节点
        self.untried_actions = [act for act in self.untried_actions if act != action]
        return new_node

    def update(self, reward):
        self.visits += 1
        alpha = 1 / self.visits
        self.value += alpha * (reward - self.value)
        if self.action is not None:
            self.action_values[self.action] += alpha * (reward - self.action_values[self.action])
        
        # 更新父节点并刷新UCB相关的缓存值
        if self.parent:
            self.parent.update(reward)
            self.parent.update_ucb_log_terms()


    def is_fully_expanded(self):
        """
        Checks if this node has expanded all possible actions.
        """
        # 检查是否所有可能的动作都有对应的子节点
        return set(self.untried_actions).issubset(self.children.keys())
    
    def mark_for_recycling(self):
        """
        Marks this node and its subtree for recycling.
        """
        self.state = None
        self.env = None
        self.parent = None
        self.action = None
        self.children = None
        self.action_values = None
        self.untried_actions = None

class MCTS:
    """
    An optimized version of the Monte Carlo Tree Search (MCTS) algorithm.
    This implementation focuses on efficiency, readability, and flexibility.
    """
    
    def __init__(self, environment, num_simulations, depth_threshold=10, recycling_threshold=1000, time_limit=500, convergence_threshold=10):
        self.environment = copy.deepcopy(environment)
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_threshold = depth_threshold  # 深度阈值
        self.recycling_threshold = recycling_threshold  # 节点回收阈值
        self.recycling_counter = 0  # 回收计数器
        self.time_limit = time_limit  # 时间限制
        self.convergence_threshold = convergence_threshold  # 收敛阈值
    
    def adaptive_num_simulations(self, node):
        """
        根据节点的重要性和不确定性动态调整模拟次数。
        """
        importance_weight = 0.5  # 重要性权重
        uncertainty_weight = 0.5  # 不确定性权重
        
        importance_score = node.importance()
        uncertainty_score = node.uncertainty()
        
        # 使用简单的线性组合来计算总分数
        score = (importance_weight * importance_score + uncertainty_weight * uncertainty_score)
        
        # 将总分数映射到模拟次数的调整上，这里使用了简单的线性关系，您可以根据需要调整
        additional_simulations = int(score / 10)  # 每10分增加一次模拟
        
        # 确保至少执行一次模拟
        return max(1, self.num_simulations + additional_simulations)
    
    def _recycle_nodes(self, node):
        if node is None:
            return
        # Recursively dispose of children nodes
        for child in list(node.children.values()):
            self._recycle_nodes(child)
        node.dispose()  # Call dispose to free resources
        self.recycling_counter += 1
        # Force garbage collection if threshold is exceeded
        if self.recycling_counter >= self.recycling_threshold:
            import gc
            gc.collect()
            self.recycling_counter = 0

    def best_child(self, node_ref, c_param=1.4, LARGE_VALUE=-float('inf'), return_default=False):
        node = node_ref
        if node is None:
            return None

        best_action = None
        best_ucb_score = LARGE_VALUE
        for action, child_node in node.children.items():
            ucb_score = node.calculate_ucb_score(child_node)
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        return node.children.get(best_action, None if not return_default else Node(node.state, self.environment))

    
    def _simulate(self, node, mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value=False):
        
        simulation_env = copy.deepcopy(self.environment)
        simulation_env.is_simulated = True
        simulation_env.set_state(node.state)
        total_reward = 0
        done = False
        
        # 直接使用value头输出一个估值，而不是实际跑完一遍模拟
        if use_mcts_vnet_value:
            global_matrix = torch.tensor(simulation_env.state['global_matrix'], dtype=torch.float).to(self.device)
            local_matrix = torch.tensor(simulation_env.state['local_matrix'], dtype=torch.float).to(self.device)
            local_matrix = local_matrix.unsqueeze(0)

            global_flow_output = global_flow_model(global_matrix)
            total_reward = mcts_vnet_model(global_flow_output, local_matrix)[1].item()
        else:
            while not done:
                global_matrix = torch.tensor(simulation_env.state['global_matrix'], dtype=torch.float).to(self.device)
                local_matrix = torch.tensor(simulation_env.state['local_matrix'], dtype=torch.float).to(self.device)
                local_matrix = local_matrix.unsqueeze(0)
                
                global_flow_output = global_flow_model(global_matrix)
                action = simulation_env.drone.sample_action(global_flow_output, local_matrix, is_greed=True)
                
                _, reward, done = simulation_env.step(action=action, global_flow_model=global_flow_model, local_flow_model=local_flow_model, is_simulated=True)
                total_reward += reward
        
        return total_reward
    
    def _search(self, state, mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value):
        self.root_node = Node(state, self.environment)
        self.root_node.update_ucb_log_terms()  # 初始化根节点的对数缓存
        num_simulations = self.adaptive_num_simulations(self.root_node)
        
        # 保存环境的初始状态
        initial_state = copy.deepcopy(self.environment.state)
        
        start_time = time.time()
        best_action = None
        best_action_stable_count = 0
        
        for _ in range(num_simulations):
            node = self.root_node

            if node.depth >= self.depth_threshold:
                break
            
            if time.time() - start_time > self.time_limit:
                break
            
            # Perform Selection
            while node and node.is_fully_expanded() and not self.environment._check_done(node.state):
                node = self.best_child(node)
            
            # Perform Expansion
            if node and not node.is_fully_expanded():
                action = node.untried_actions[0]
                new_state, reward, done = self.environment.step(global_flow_model=global_flow_model, local_flow_model=local_flow_model, action=action, is_simulated=True)
                # 恢复环境的初始状态
                self.environment.set_state(initial_state)
                if not done:
                    node = node.add_child(new_state, action)
            
            # Perform Simulation
            if node and not self.environment._check_done(node.state):
                reward = self._simulate(node=node, mcts_vnet_model=mcts_vnet_model, global_flow_model=global_flow_model, local_flow_model=local_flow_model, use_mcts_vnet_value=use_mcts_vnet_value)
            
                # Perform Backpropagation
                while node is not None:
                    current_node = node
                    if current_node is None:
                        break
                    current_node.update(reward)
                    node = current_node.parent
            
            # Check if the best action is stable
            best_child_node = self.best_child(self.root_node, return_default=False)
            current_best_action = best_child_node.action if best_child_node is not None else None
            if current_best_action == best_action:
                best_action_stable_count += 1
                if best_action_stable_count >= self.convergence_threshold:
                    break
            else:
                best_action = current_best_action
                best_action_stable_count = 0
                
        if best_action is None and self.root_node.untried_actions:
            best_action = self.root_node.untried_actions[0]
        else:
            # 所有动作都已尝试，选择具有最高价值的子节点对应的动作
            best_child_node = max(self.root_node.children.values(), key=lambda child: child.value, default=None)
            best_action = best_child_node.action if best_child_node is not None else None

        # After search, clean up all nodes to prevent memory leak
        self._recycle_nodes(self.root_node)
        self.root_node = None  # Remove reference to root node
        
        return best_action

    def search(self, initial_state, mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value=False):
        """
        Public method to initiate the MCTS search given the initial state and models.
        Returns the action selected by the MCTS algorithm.
        """
        return self._search(state=initial_state, mcts_vnet_model=mcts_vnet_model, global_flow_model=global_flow_model, local_flow_model=local_flow_model, use_mcts_vnet_value=use_mcts_vnet_value)