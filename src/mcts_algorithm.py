import numpy as np
import copy
from collections import defaultdict

import torch

def deepcopy_env(env):
    new_env = copy.copy(env)  # 创建一个浅拷贝
    if hasattr(new_env, 'state_dict'):
        new_env.state_dict = copy.deepcopy(env.state_dict)
    # 如果还有其他需要深拷贝的属性，继续添加类似的代码
    return new_env

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self):
        return len(self.children) == 9

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class MCTS:
    def __init__(self, environment, num_simulations):
        self.environment = environment
        self.num_simulations = num_simulations

    def simulate(self, node, global_flow_model, local_flow_model):
        # Perform a rollout from the given node
        current_env = copy.deepcopy(self.environment)
        current_env.is_simulated=True
        current_env.set_state(node.state)
        total_reward = 0
        while not current_env._check_done():

            global_matrix = current_env.state['global_matrix']
            # print(current_env.state)
            local_matrix = current_env.state['local_matrix']
            # print(global_matrix)
            global_matrix = torch.tensor(global_matrix, dtype=torch.float)
            global_matrix = global_matrix.view(1, -1)  # Reshape to (1, seq_len * embedding_dim)
            local_matrix = torch.tensor(local_matrix, dtype=torch.float32)

            # print(local_matrix.shape)
            global_flow_output = global_flow_model(global_matrix)
            local_flow_output = local_flow_model(local_matrix)

            action = current_env.drone.sample_action(global_flow_output, local_flow_output)
            # print(action)
            _, reward, _ = current_env.step(action=action, global_flow_model=global_flow_model, local_flow_model=local_flow_model, is_simulated=True)
            total_reward += reward
        return total_reward


    def search(self, global_flow_model, local_flow_model):
        root_state = self.environment.reset()
        root_node = Node(root_state)

        for _ in range(self.num_simulations):
            node = root_node
            current_env = copy.deepcopy(self.environment)
            current_env.is_simulated=True
            current_env.set_state(root_state)

            # Selection
            while node.is_fully_expanded() and not current_env._check_done():
                node = node.best_child()
                current_env.set_state(node.state)

            # Expansion
            if not node.is_fully_expanded() and not current_env._check_done():
                global_matrix = current_env.state['global_matrix']
                local_matrix = current_env.state['local_matrix']

                global_matrix = torch.tensor(global_matrix, dtype=torch.float)
                global_matrix = global_matrix.view(1, -1)  # Reshape to (1, seq_len * embedding_dim)
                local_matrix = torch.tensor(local_matrix, dtype=torch.float32)

                
                global_flow_output = global_flow_model(global_matrix)

                local_flow_output = local_flow_model(local_matrix)

                action = current_env.drone.sample_action(global_flow_output, local_flow_output)
                new_state, reward, _ = current_env.step(action=action, global_flow_model=global_flow_model, local_flow_model=local_flow_model)
                new_node = Node(new_state, node, action)
                node.add_child(new_node)
                node = new_node

            # Simulation
            reward = self.simulate(node, global_flow_model, local_flow_model)

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        return root_node.best_child().action