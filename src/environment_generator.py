import copy
import numpy as np
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

        self.mcts = MCTS(self, num_simulations=0)
        self.state = None
        self.reward = 0
        self.done = False
        self.drone = Drone(self.config)  # 创建 Drone 实例
        self.data_collector = DataCollector()
        self.is_simulated = False
        self.num_signal_points = 0
        self.num_obstacles = 0

    def set_state(self, state):
        self.state = state
        self.drone_position = state['drone_position']
        self.drone.battery = state['battery']

    def reset(self):
        environment = self._generate_initial_environment()
        if self.state is None:
            self.state = {
                'global_matrix': environment,
                'drone_position': None,
                'local_matrix': None,
                'battery': None
            }
        self.drone.reset(self._is_valid_position)  # 重置无人机状态
        self.reward = 0
        self.done = False
        self.data_collector.reset()

        # 构建初始状态字典
        self.state = {
            'global_matrix': environment,
            'drone_position': self.drone.position,
            'local_matrix': self._get_local_matrix(),
            'battery': self.drone.battery
        }

        return self.state

    def run_simulation_animation(self):
        self.visualizer = SimulationVisualizer(self)  # 将当前环境实例传给可视化器
        self.visualizer.start_animation()

    def step(self, global_flow_model, local_flow_model, action=None, use_mcts=False, is_simulated=False, use_mcts_to_train=False):
        
        # 在线学习时
        action_index = action
        action = self.drone.available_actions[action_index]

        if self.done:
            # 如果环境已完成，直接返回当前状态、奖励和完成标志
            return self.state, self.reward, self.done

        if use_mcts:
            # 使用 MCTS 算法来获取最佳动作
            action = self.mcts.search(global_flow_model, local_flow_model)

        # 执行动作并更新环境状态
        self._update_drone_position(action)
        self._update_state()
        self._check_done()

        # 构建当前状态字典
        state_dict = {
            'global_matrix': self.state['global_matrix'],
            'drone_position': self.drone.position,
            'local_matrix': self.state['local_matrix'],
            'battery': self.drone.battery
        }

        # 使用 MCTS 模拟从当前状态出发的一系列动作，并获取累积回报，当已经在模拟时则跳过
        if not is_simulated and use_mcts_to_train:
            cumulative_reward = self.mcts.simulate(Node(state_dict), global_flow_model=global_flow_model, local_flow_model=local_flow_model)

            # 将当前状态和累积回报作为一对数据添加到 DataCollector 中
            # self.data_collector.collect(state_dict, cumulative_reward)

        # 展示当前状态
        # print(f"State: {state_dict}, Reward: {self.reward}, Done: {self.done}")

        return state_dict, self.reward, self.done

    def _generate_initial_environment(self):
        size = self.config['environment']['size']
        max_num_signal_points = self.config['environment']['max_num_signal_points']
        max_num_obstacles = self.config['environment']['max_num_obstacles']
        min_num_signal_points = self.config['environment']['min_num_signal_points']
        min_num_obstacles = self.config['environment']['min_num_obstacles']

        # Initialize the environment with all zeros
        environment = np.zeros((size, size))

        # Randomly generate the number of signal points and obstacles
        num_signal_points = np.random.randint(min_num_signal_points, max_num_signal_points + 1)
        self.num_signal_points = num_signal_points

        num_obstacles = np.random.randint(min_num_obstacles, max_num_obstacles + 1)
        self.num_obstacles = num_obstacles

        # Randomly generate positions for signal points and obstacles
        # Ensure that they do not overlap
        all_positions = set()
        while len(all_positions) < num_signal_points + num_obstacles:
            new_position = tuple(np.random.randint(0, size, 2))
            all_positions.add(new_position)

        # Assign signal points and obstacles to the environment
        for i, position in enumerate(all_positions):
            if i < num_signal_points:
                environment[position] = 1  # Signal point
            else:
                environment[position] = -1  # Obstacle

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
        self.reward += reward  # 更新累积奖励

    def _update_state(self):

        # 随着时间耗电
        self.drone.battery -= self.config['penalties_rewards']["default_battery_penalty"]
        self.state["battery"] = self.drone.battery
        
        # 更新局部矩阵，如果有必要的话
        self.state['local_matrix'] = self._get_local_matrix()

    def _get_local_matrix(self):
        # 使用 Drone 类的方法来获取局部矩阵
        return self.drone.get_local_matrix(self.state, self.config['environment']['perception_range'])

    def _calculate_reward(self, action_result):
        # Define the rewards for different action results
        rewards = {
            'collect_success': self.config['penalties_rewards']["target_reward"] + self.config['penalties_rewards']["default_penalty"],  # Reward for successfully collecting an information point
            'collect_fail': self.config['penalties_rewards']["invalid_action_penalty"] + self.config['penalties_rewards']["default_penalty"],     # Penalty for failing to collect an information point
            'invalid_move': self.config['penalties_rewards']["invalid_action_penalty"] + self.config['penalties_rewards']["default_penalty"],      # Penalty for an invalid move
            'valid_move': self.config['penalties_rewards']["default_penalty"]               # Penalty for a valid move
        }

        # 简易版本的信号点计算
        if action_result == 'collect_success':
            self.num_signal_points -= 1

        return rewards.get(action_result, 0)

    def _check_done(self):
        # Check if the drone's battery is depleted
        if self.drone.battery <= 0 or self.num_signal_points == 0:
            # self.data_collector.save_data(self.mcts_path)
            self.done = True

            # 展示当前状态
            print(f"State: {self.state}, Reward: {self.reward}, Done: {self.done}")
        else:
            self.done = False
        
        return self.done

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
