import random
import numpy as np
import torch
from torch.distributions import Categorical

class Drone:
    def __init__(self, config, mcts_vnet_model=None):
        self.config = config
        self.position = None
        self.battery = config['drone']['battery']
        self.collisions = 0
        self.max_collect_range = config['drone']['max_collect_range']
        self.available_actions = self._generate_available_actions()[0]
        self.mcts_vnet_model = mcts_vnet_model  # 将 MCTSVNet 模型传入无人机类
        self.online_learning = self.config["training"]["online_learning"]
        self.online_train_seed = self.config["environment"]["online_train_seed"]
        self.test_seed = self.config["environment"]["test_seed"]
        
        if self.online_learning:
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
            torch.manual_seed(self.test_seed)
            torch.cuda.manual_seed_all(self.test_seed)  # 如果使用多个CUDA设备

    def reset(self, is_valid_position):
        self.position = self._find_initial_position(is_valid_position)
        self.battery = self.config['drone']['battery']
        self.collisions = 0

    def _find_initial_position(self, is_valid_position):
    # Check if the initial position is specified in the config
        initial_position = self.config['drone'].get('initial_position', None)
        if initial_position is not None:
            # Validate the initial position
            if is_valid_position(initial_position):
                return initial_position
            else:
                raise ValueError("Invalid drone initial position specified in the config.")
        else:
            # Randomly generate a valid initial position for the drone
            while True:
                x = np.random.randint(0, self.config['environment']['size'])
                y = np.random.randint(0, self.config['environment']['size'])
                if is_valid_position((x, y)):
                    return (x, y)

    def update_position(self, action, state, is_valid_position, calculate_reward):
        current_position = self.position
        action_type, action_value = action

        # Handle the move action
        if action_type == 'move':
            new_position = (current_position[0] + action_value[0], current_position[1] + action_value[1])
            if is_valid_position(new_position):
                self.position = new_position
                # Assume moving costs battery life; adjust the cost accordingly
                self.battery -= self.config['penalties_rewards']["move_cost"]
                reward = calculate_reward('valid_move')
            else:
                self.collisions += 1
                reward = calculate_reward('invalid_move')

        # Handle the collect action
        elif action_type == 'collect':
            collected = False
            max_collect_range = self.config['drone']['max_collect_range']
            
            for dx in range(-max_collect_range, max_collect_range + 1):
                for dy in range(-max_collect_range, max_collect_range + 1):
                    check_position = (current_position[0] + dx, current_position[1] + dy)
                    if is_valid_position(check_position) and state['global_matrix'][check_position[0], check_position[1]] == 1:
                        state['global_matrix'][check_position[0], check_position[1]] = 0
                        self.battery -= self.config['penalties_rewards']["collect_info_cost"]
                        collected = True
                        # Update the collected points counter if available
                        state['collected_points'] = state.get('collected_points', 0) + 1
                        break
                if collected:
                    break

            reward = calculate_reward('collect_success') if collected else calculate_reward('collect_fail')

        # 默认扣除agent电量
        self.battery -= self.config['penalties_rewards']["default_battery_penalty"]
        return reward


    def get_local_matrix(self, state, perception_range):
        # 使用正确的属性和参数
        matrix_size = 2 * perception_range + 1
        drone_position = np.array(self.position, dtype=int)

        # 初始化局部矩阵为 -1
        local_matrix = np.full((matrix_size, matrix_size), -1)

        # 计算局部矩阵的边界
        min_x = max(0, drone_position[0] - perception_range)
        max_x = min(state['global_matrix'].shape[0], drone_position[0] + perception_range + 1)
        min_y = max(0, drone_position[1] - perception_range)
        max_y = min(state['global_matrix'].shape[1], drone_position[1] + perception_range + 1)

        # 填充局部矩阵
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                local_x = x - drone_position[0] + perception_range
                local_y = y - drone_position[1] + perception_range
                local_matrix[local_x, local_y] = state['global_matrix'][x, y]

        return local_matrix


    def _generate_available_actions(self):
        max_speed = self.config['drone']['max_speed']
        # 生成所有可能的移动动作，考虑最大速度限制
        moves = [(dx, dy) for dx in range(-max_speed, max_speed + 1) for dy in range(-max_speed, max_speed + 1) if (dx, dy) != (0, 0)]
        actions = [('move', move) for move in moves] + [('collect', None)]
        num_action = len([('move', move) for move in moves] + [('collect', None)])
        
        # 动作字典1，用于将动作转换为索引
        self.available_actions_dict_inv = {action: i for i, action in enumerate(actions)}
        # 动作字典2，用于将索引转换为动作
        self.available_actions_dict = {i: action for i, action in enumerate(actions)}
        
        return actions, num_action


    def sample_action(self, global_flow_output, local_flow_output, is_greed=True):

        # 根据全局流和局部流模型的输出选择动作
        policy_output, _ = self.mcts_vnet_model(global_flow_output, local_flow_output)  # 忽略价值输出
        if is_greed:
            policy_output_np = policy_output.detach().cpu().numpy()  # 转换为NumPy数组，确保操作在CPU上进行
            action = np.argmax(policy_output_np)  # 选择概率最高的动作
            return action
        else:
            action_dist = Categorical(policy_output)
            action = action_dist.sample()
            return action

