import copy
import random
import numpy as np
import torch
import yaml
import sys
import os

# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_collector import DataCollector
from src.mcts_algorithm import MCTS, Node
from entities.drone import Drone
from utils.visualizer import SimulationVisualizer  # 导入 Drone 类

class Environment:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.online_learning = self.config["training"]["online_learning"]
        self.online_train_seed = self.config["environment"]["online_train_seed"]
        self.test_seed = self.config["environment"]["test_seed"]
        self.state = None
        self.drone = Drone(self.config)  # 创建 Drone 实例
        self.data_collector = DataCollector()
        self.is_simulated = False
        self.num_obstacles = 0
        self.episode_count = 0
        self.initial_environment = self._generate_initial_environment()  # 初始环境的副本
        self.max_num_signal_points = 0
        
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
            torch.manual_seed(self.online_train_seed)
            torch.cuda.manual_seed_all(self.online_train_seed)  # 如果使用多个CUDA设备

    def set_state(self, state):
        self.state = state
        self.drone.battery = state['battery']
        self.drone.position = state['drone_position']

    def reset(self):
        if self.online_learning:
            # 在线学习时，基于episode_count调整环境
            if self.episode_count % self.config['environment']['reset_interval'] == 0:
                # 周期性重置环境
                self.initial_environment = self._generate_initial_environment()
            elif self.episode_count % self.config['environment']['adjust_interval'] == 0:
                # 微调环境
                self.micro_adjust_environment()
            self.state = {
                'global_matrix': copy.deepcopy(self.initial_environment),
                'drone_position': None,
                'local_matrix': None,
                'battery': None,
                'collected_points': 0,
                'exist_signal_points': 0,
                'action':None,
                'reward': 0,
                'accumulated_reward': 0,
                'done': False
            }

        else:
            environment = self._generate_initial_environment()
            if self.state is None:
                self.state = {
                    'global_matrix': environment,
                    'drone_position': None,
                    'local_matrix': None,
                    'battery': None,
                    'collected_points': 0,
                    'exist_signal_points': 0,
                    'action':None,
                    'reward': 0,
                    'accumulated_reward': 0,
                    'done': False
                }
            
        # 共用的重置逻辑
        self.drone.reset(self._is_valid_position)
        # self.data_collector.reset()
        self.state['drone_position'] = self.drone.position
        self.state['local_matrix'] = self._get_local_matrix()
        self.state['battery'] = self.drone.battery
        self.state['exist_signal_points'] = np.sum(self.state['global_matrix'] == 1)
        self.max_num_signal_points = self.state['exist_signal_points']
        self.episode_count += 1
        
        return self.state

    def run_simulation_animation(self):
        self.visualizer = SimulationVisualizer(self)  # 将当前环境实例传给可视化器
        self.visualizer.start_animation()

    def step(self, global_flow_model, local_flow_model, mcts_vnet_model=None, action=None, use_mcts=False, is_simulated=False, use_mcts_to_train=False, use_mcts_vnet_value=False):
        
        self._check_done(None, is_simulated)
        if self.state['done']:
            # 如果环境已完成，直接返回当前状态、奖励和完成标志
            return self.state, self.state['reward'], self.state['done']

        if use_mcts and not is_simulated:
            mcts = MCTS(copy.deepcopy(self), num_simulations=self.config['mcts']['num_simulations'])
            # 使用 MCTS 算法来获取最佳动作
            action = mcts.search(copy.deepcopy(self.state), mcts_vnet_model, global_flow_model, local_flow_model, use_mcts_vnet_value)
            del mcts  # 删除 MCTS 实例，以便下一次重新创建

        if type(action) != tuple:
            action = self.drone.available_actions_dict[action]

        # 执行动作并更新环境状态
        self._update_drone_position(action)
        self._update_state()
        self._check_done(None, is_simulated)

        # 构建当前状态字典
        state_dict = {
            'global_matrix': self.state['global_matrix'],
            'drone_position': self.drone.position,
            'local_matrix': self.state['local_matrix'],
            'battery': self.drone.battery,
            'collected_points': self.state['collected_points'],
            'action': action,
            'reward': self.state['reward'],
            'accumulated_reward':self.state['accumulated_reward'],
            'done': self.state['done']
        }
        state_dict['exist_signal_points'] = np.sum(self.state['global_matrix'] == 1)

        # 使用 MCTS 模拟从当前状态出发的一系列动作，并获取累积回报，当已经在模拟时则跳过
        if not is_simulated and use_mcts_to_train:
            mcts = MCTS(copy.deepcopy(self), num_simulations=self.config['mcts']['num_simulations'])
            cumulative_reward = mcts.simulate(Node(state_dict), global_flow_model=global_flow_model, local_flow_model=local_flow_model)

        # 展示当前状态
        # print(f"State: {state_dict}")

        return state_dict, state_dict['reward'], state_dict['done']

    def micro_adjust_environment(self):
        size = self.config['environment']['size']
        adjustment_factor = self.config['environment']['adjustment_factor']  # 微调的比例因子
        
        # 创建环境的深拷贝，以避免在原始环境上直接修改
        temp_environment = copy.deepcopy(self.initial_environment)
        
        # 计算需要调整的信号点和障碍物数量
        num_adjustments = max(1, int((self.state['exist_signal_points'] + self.num_obstacles) * adjustment_factor))
        
        # 随机选择信号点和障碍物进行移动
        for _ in range(num_adjustments):
            # 重新随机选择一个点直到这个点是信号点或障碍物
            while True:
                old_position = tuple(np.random.randint(0, size, 2))
                if temp_environment[old_position[0], old_position[1]] != 0:
                    break
            
            # 寻找新位置
            while True:
                new_position = tuple(np.random.randint(0, size, 2))
                # 确保新位置不与其他信号点或障碍物重叠
                if temp_environment[new_position[0], new_position[1]] == 0:
                    # 将信号点或障碍物移动到新位置
                    temp_environment[new_position[0], new_position[1]] = temp_environment[old_position[0], old_position[1]]
                    temp_environment[old_position[0], old_position[1]] = 0
                    break
        
        # 随机添加或删除信号点/障碍物
        if np.random.rand() < 0.5:  # 50%的概率进行添加或删除操作
            modify_type = np.random.choice([-1, 1])  # 随机选择添加信号点(1)或添加障碍物(-1)
            modify_position = tuple(np.random.randint(0, size, 2))
            # 只有在选中的位置为空时才进行添加
            if temp_environment[modify_position[0], modify_position[1]] == 0:
                temp_environment[modify_position[0], modify_position[1]] = modify_type
            elif modify_type == temp_environment[modify_position[0], modify_position[1]]:
                # 如果随机选择的位置已经是对应类型，则尝试删除
                temp_environment[modify_position[0], modify_position[1]] = 0

        # 更新初始环境
        self.initial_environment = temp_environment
        self.state['exist_signal_points'] = np.sum(temp_environment == 1)

    def _generate_initial_environment(self):
        size = self.config['environment']['size']
        complexity_factor = min(1.0, (self.episode_count / self.config['environment']['complexity_interval'])) if self.online_learning else 1.0
        
        # 保持最小值不变，只对最大值应用复杂度因子
        min_num_signal_points = self.config['environment']['min_num_signal_points']
        min_num_obstacles = self.config['environment']['min_num_obstacles']
        # 应用复杂度因子后确保最大值不小于最小值
        max_num_signal_points = max(min_num_signal_points, int(self.config['environment']['max_num_signal_points'] * complexity_factor))
        max_num_obstacles = max(min_num_obstacles, int(self.config['environment']['max_num_obstacles'] * complexity_factor))

        environment = np.zeros((size, size))
        num_signal_points = np.random.randint(min_num_signal_points, max_num_signal_points + 1)
        num_obstacles = np.random.randint(min_num_obstacles, max_num_obstacles + 1)
        self.max_num_signal_points = num_signal_points # 用于计算效率奖励
        self.num_obstacles = num_obstacles

        all_positions = set()
        attempt_counter = 0
        while len(all_positions) < num_signal_points + num_obstacles and attempt_counter < size ** 2:
            new_position = tuple(np.random.randint(0, size, 2))
            if new_position not in all_positions:
                all_positions.add(new_position)
            attempt_counter += 1

        # 为了保证环境的初始化总是成功，我们在循环外再次检查数量并作出调整
        for position in all_positions:
            if num_signal_points > 0:
                environment[position] = 1
                num_signal_points -= 1
            else:
                environment[position] = -1
        
        self.num_obstacles = np.sum(environment == -1)

        return environment
                
    def _is_valid_position(self, position):
    # Check if the position is within the bounds and not on an obstacle
        x, y = position
        if 0 <= x < self.config['environment']['size'] and 0 <= y < self.config['environment']['size']:
            return self.state['global_matrix'][x, y] != -1  # Not an obstacle
        return False

    def _update_drone_position(self, action):
        # 使用 Drone 类的方法来更新无人机位置
        reward = self.drone.update_position(action, self.state, self._is_valid_position, self._calculate_reward)
        self.state['exist_signal_points'] = np.sum(self.state['global_matrix'] == 1)
        self.state['reward'] = reward  # 更新奖励
        self.state['accumulated_reward'] += reward  # 累积奖励

    def _update_state(self):
        
        self.state['battery'] = self.drone.battery
        self.state['drone_position'] = self.drone.position
        self.state['local_matrix'] = self._get_local_matrix()
        self.state['exist_signal_points'] = np.sum(self.state['global_matrix'] == 1)

    def _get_local_matrix(self):
        # 使用 Drone 类的方法来获取局部矩阵
        return self.drone.get_local_matrix(self.state, self.config['environment']['perception_range'])

    # 奖励函数设计模块

    def calculate_efficiency_reward(self):
        progress = self.state['collected_points'] / self.max_num_signal_points
        smooth_factor = min(1, progress * 2)  # 当进度小于50%时，平滑效率奖励
        efficiency_reward = (self.state['battery'] / self.config['drone']['battery']) * self.config['penalties_rewards']['efficiency_multiplier'] * progress * smooth_factor
        return efficiency_reward

    def calculate_dynamic_reward(self, action_result):
        # 动态调整奖励
        if action_result == 'collect_success':
            collected_points = self.state['collected_points']
            # 随着收集到的信息点增加，奖励递增
            dynamic_reward = self.config['penalties_rewards']["target_reward"] * (collected_points / self.max_num_signal_points)
        else:
            dynamic_reward = 0
        return dynamic_reward

    def calculate_long_term_reward(self):
        if self.state['collected_points'] > 0:
            efficiency = self.state['collected_points'] / (self.config['drone']['battery'] - self.state['battery'] + 1)  # 避免除以零
            long_term_reward = min(efficiency, self.config['penalties_rewards']['max_efficiency']) * self.config['penalties_rewards']['long_term_multiplier']
        else:
            long_term_reward = 0
        return long_term_reward

    def _calculate_reward(self, action_result):
        # 定义不同动作结果的奖励
        rewards = {
            'collect_success': self.config['penalties_rewards']["target_reward"],
            'collect_fail': -self.config['penalties_rewards']["invalid_action_penalty"],
            'invalid_move': -self.config['penalties_rewards']["collision_penalty"],
            'valid_move': -self.config['penalties_rewards']["default_penalty"]
        }
        reward = rewards.get(action_result, 0)

        # 添加效率奖励、动态奖励和长期奖励
        efficiency_reward = self.calculate_efficiency_reward()
        dynamic_reward = self.calculate_dynamic_reward(action_result)
        long_term_reward = self.calculate_long_term_reward()

        return reward + efficiency_reward + dynamic_reward + long_term_reward

    # 结束模块

    def _check_done(self, state=None, is_simulated=False):
        if state is None:
            # 增加判断done的条件
            if self.state['battery'] < 0.1 or self.state['exist_signal_points'] == 0:
                self.state['done'] = True
                if not is_simulated:
                    # 展示实际终局状态
                    print(f"State: {self.state}")
                self.state['accumulated_reward']=0
            else:
                self.state['done'] = False
            return self.state['done']
        else:
            if state['battery'] < 0.1 or self.state['exist_signal_points'] == 0:
                self.state['accumulated_reward']=0
                return True
            return False

if __name__ == "__main__":
    config_path = 'config/config.yml'
    env = Environment(config_path)
    state = env.reset()
    done = False
    while not done:
        action_type = np.random.choice(["move", "collect"])
        action_value = (np.random.randint(-1, 2), np.random.randint(-1, 2))
        action = (action_type, action_value)
        state, reward, done = env.step(action=action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
