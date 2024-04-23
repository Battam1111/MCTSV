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
        self.untried_actions = set(env.drone._generate_available_actions()[0]) if parent is None else parent.untried_actions.copy()
        self.c_param = c_param  # 初始探索参数
        self.ucb_score = float('inf')
        self.total_log_parent_visits = np.log(1)

    def calculate_ucb_score(self, child):
        """
        根据当前的访问次数和探索参数动态计算UCB分数。
        """
        if child.visits == 0:
            return float('inf')  # 未访问的节点优先探索

        self.update_parent_visits_log()  # 更新缓存
        exploitation = child.value / child.visits
        exploration = self.c_param * np.sqrt(self.total_log_parent_visits / child.visits)
        return exploitation + exploration

    def dynamic_c_param(self):
        """
        动态调整c_param基于节点深度和其他可能因素。
        """
        # 示例：随深度增加而减少c_param以鼓励更深层的利用
        return max(0.1, self.c_param * 0.95 ** self.depth)

    def update(self, reward):
        """
        使用动态调整的alpha更新节点的价值。
        """
        self.visits += 1
        alpha = 1 / self.visits
        self.value += alpha * (reward - self.value)
        if self.action is not None:
            self.action_values[self.action] += alpha * (reward - self.action_values[self.action])

        # 动态调整探索参数
        if self.parent:
            self.parent.update(reward)
            self.parent.c_param = self.parent.dynamic_c_param()  # 更新探索参数
            self.parent.update_parent_visits_log()

    def add_child(self, state, action):
        """
        Adds a new child node for the given state and action.
        Optimized to manage untried actions using set for efficiency.
        """
        new_node = Node(state, self.env, self, action, self.depth + 1)
        self.children[action] = new_node
        # 优化：使用集合移除动作，更高效
        self.untried_actions.discard(action)
        return new_node

    def is_fully_expanded(self):
        """
        Checks if this node has expanded all possible actions.
        Optimized by checking if untried_actions is empty.
        """
        return not self.untried_actions

    def update_parent_visits_log(self):
        if self.parent:
            # 确保每次计算UCB分数前父节点的访问次数对数是最新的
            self.total_log_parent_visits = np.log(self.parent.visits + 1)

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
    

class MCTS:
    """
    An optimized version of the Monte Carlo Tree Search (MCTS) algorithm.
    This implementation focuses on efficiency, readability, and flexibility.
    """
    
    def __init__(self, environment, num_simulations, depth_threshold=10, time_limit=500, convergence_threshold=10):
        self.environment = environment
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_threshold = depth_threshold  # 深度阈值
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

    def best_child(self, node, LARGE_VALUE=-float('inf'), return_default=True):
        """
        Enhanced to handle edge cases where no children exist but a default action is necessary.
        """
        if node is None or not node.children:
            if return_default:
                # 提供一个默认行动，可能是重复最后一个行动或一个特殊的行动
                last_action = node.parent.action if node.parent else None
                return last_action  # 可以设定为一个特定的默认行动
            else:
                return None

        best_action = None
        best_ucb_score = LARGE_VALUE
        for action, child_node in node.children.items():
            ucb_score = node.calculate_ucb_score(child_node)
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        return node.children.get(best_action)
    
    def _simulate(self, node, mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value=False):
        # 保存环境的初始状态
        initial_state = copy.deepcopy(self.environment.state)

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
        
            # 恢复环境的初始状态
            self.environment.set_state(initial_state)
        
        return total_reward
    
    def _search(self, state, mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value):
        self.root_node = Node(state, self.environment)
        num_simulations = self.adaptive_num_simulations(self.root_node)

        initial_state = copy.deepcopy(self.environment.state)
        start_time = time.time()
        best_action = None
        best_action_stable_count = 0

        for simulation_index in range(num_simulations):
            node = self.root_node  # 总是从根节点开始
            path = [node]  # 初始化路径

            # Selection: 寻找最佳子节点直至叶子节点
            while node.is_fully_expanded() and node.children:
                node = self.best_child(node)
                if node is None or self.environment._check_done(node.state):
                    break  # 如果没有子节点可选或已达终止状态，则跳出循环
                path.append(node)

            # Expansion: 如果节点未完全扩展且未达到结束状态，则创建新的子节点
            if node and not node.is_fully_expanded() and not self.environment._check_done(node.state):
                action = node.untried_actions.pop()
                new_state, reward, done = self.environment.step(global_flow_model=global_flow_model, local_flow_model=local_flow_model, action=action, is_simulated=True)
                self.environment.set_state(initial_state)  # 恢复环境的初始状态
                if not done:
                    node = node.add_child(new_state, action)
                    path.append(node)
                # 最终状态直接返回action作为best_action,不然会报错
                else:
                    best_action = action

            # Simulation: 从当前节点模拟直到游戏结束
            reward = 0
            if node and not self.environment._check_done(node.state):
                reward = self._simulate(node=node, mcts_vnet_model=mcts_vnet_model, global_flow_model=global_flow_model, local_flow_model=local_flow_model, use_mcts_vnet_value=use_mcts_vnet_value)
            
            # Backpropagation: 回传更新所有经过的节点
            for node in reversed(path):
                node.update(reward)
                if node.parent:
                    node.parent.update_parent_visits_log()

            # 检查最佳动作是否稳定
            if self.root_node.children:
                best_child_node = self.best_child(self.root_node, return_default=True)
                current_best_action = best_child_node.action if best_child_node else None
                if current_best_action == best_action:
                    best_action_stable_count += 1
                    if best_action_stable_count >= self.convergence_threshold:
                        break
                else:
                    best_action = current_best_action
                    best_action_stable_count = 0

            # 检查时间限制
            if time.time() - start_time > self.time_limit:
                break
        
        return best_action


    def search(self, initial_state, mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value=False):
        """
        Public method to initiate the MCTS search given the initial state and models.
        Returns the action selected by the MCTS algorithm.
        """
        return self._search(state=initial_state, mcts_vnet_model=mcts_vnet_model, global_flow_model=global_flow_model, local_flow_model=local_flow_model, use_mcts_vnet_value=use_mcts_vnet_value)
