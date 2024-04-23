import json
import numpy as np
import yaml
import pandas as pd
import os

class GlobalFlowDataPreparer:
    def __init__(self, config_path):
        # 载入配置文件
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def generate_environment(self):
        size = self.config['environment']['size']
        max_num_signal_points = self.config['environment']['max_num_obstacles']
        max_num_obstacles = self.config['environment']['max_num_obstacles']
        min_num_signal_points = self.config['environment']['min_num_signal_points']
        min_num_obstacles = self.config['environment']['min_num_obstacles']

        # 随机生成信号点和障碍物的数量
        num_signal_points = np.random.randint(min_num_signal_points, max_num_signal_points + 1)
        num_obstacles = np.random.randint(min_num_obstacles, max_num_obstacles + 1)

        # 一次性随机生成所有点的位置
        all_points = set()
        while len(all_points) < num_signal_points + num_obstacles:
            point = tuple(np.random.randint(0, size, 2))
            all_points.add(point)

        # 分配点给信号点和障碍物
        all_points = np.array(list(all_points))
        signal_points = all_points[:num_signal_points]
        obstacle_points = all_points[num_signal_points:num_signal_points + num_obstacles]

        return signal_points, obstacle_points

    
    def value_function(self, signal_points, obstacle_points, position):
        signal_weight = self.config['environment']['signal_weight']
        obstacle_weight = self.config['environment']['obstacle_weight']
        obstacle_penalty = self.config['environment']['obstacle_penalty']
        
        # 确保position为二维向量以计算距离
        position = np.array(position)  # 确保position是数组格式

        signal_contribution = sum(np.exp(-np.linalg.norm(signal - position, axis=-1)) for signal in signal_points)
        obstacle_contribution = sum(np.exp(-np.linalg.norm(obstacle - position, axis=-1)) for obstacle in obstacle_points if not np.array_equal(obstacle, position))
        if any(np.array_equal(obstacle, position) for obstacle in obstacle_points):
            return obstacle_penalty
        value_score = signal_weight * signal_contribution - obstacle_weight * obstacle_contribution
        return value_score

    
    def generate_value_matrix(self):
        size = self.config['environment']['size']
        resolution = self.config['environment']['resolution']
        signal_points, obstacle_points = self.generate_environment()
        value_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                position = np.array([i, j]) / resolution
                value_matrix[i, j] = self.value_function(signal_points, obstacle_points, position)
        return value_matrix
    
    def save_environment_to_json(self, signal_points, obstacle_points, size):
        environment_matrix = np.zeros((size, size))
        for point in signal_points:
            environment_matrix[point[0], point[1]] = 1
        for point in obstacle_points:
            environment_matrix[point[0], point[1]] = -1
        return environment_matrix.tolist()

    def save_value_matrix_to_csv(self, value_matrix):
        return pd.DataFrame(value_matrix).values.tolist()

    def batch_generate_environments(self, num_environments, output_file):
        all_data = {}
        for i in range(num_environments):
            signal_points, obstacle_points = self.generate_environment()
            value_matrix = self.generate_value_matrix()
            
            environment_data = self.save_environment_to_json(signal_points, obstacle_points, self.config['environment']['size'])
            value_matrix_data = self.save_value_matrix_to_csv(value_matrix)
            
            all_data[f'environment_{i}'] = {
                'environment': environment_data,
                'value_matrix': value_matrix_data
            }
            
            print(f'Generated environment {i} data.')

        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=4)
        print(f'All environments and value matrices have been saved to {output_file}.')


class LocalFlowDataPreparer:
    def __init__(self, combined_data_path, config_path, environment_index):
        self.combined_data = self.load_combined_data(combined_data_path)
        self.config = self.load_config(config_path)
        self.global_environment_matrix = self.extract_global_environment(environment_index)
        
    def load_combined_data(self, path):
        with open(path, 'r') as file:
            return json.load(file)
        
    def extract_global_environment(self, index):
        environment_key = f'environment_{index}'
        if environment_key in self.combined_data:
            return np.array(self.combined_data[environment_key]['environment'])
        else:
            raise ValueError(f'Environment {index} not found in the combined data file.')
        
    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def local_value_function(self, local_signals_matrix, local_obstacles_matrix, evaluated_position, drone_position):
        signal_weight = self.config['signal_weight']
        obstacle_weight = self.config['obstacle_weight']
        obstacle_penalty = self.config['obstacle_penalty']
        distance_attenuation_factor = self.config['distance_attenuation_factor']

        # 确保evaluated_position和drone_position为二维向量以计算距离
        evaluated_position = np.array(evaluated_position)
        drone_position = np.array(drone_position)

        # 计算从无人机位置到评估位置的距离
        distance_to_evaluated_position = np.linalg.norm(evaluated_position - drone_position)

        # 计算信号贡献和障碍物贡献
        signal_points = np.argwhere(local_signals_matrix == 1)
        obstacle_points = np.argwhere(local_obstacles_matrix == -1)

        if signal_points.size > 0:
            signal_contribution = sum(np.exp(-np.linalg.norm(signal_points - evaluated_position, axis=-1)))
        else:
            signal_contribution = 0

        if obstacle_points.size > 0:
            obstacle_contribution = sum(np.exp(-np.linalg.norm(obstacle_points - evaluated_position, axis=-1)))
        else:
            obstacle_contribution = 0

        local_signals_matrix = np.array(local_signals_matrix)
        local_obstacles_matrix = np.array(local_obstacles_matrix)

        if local_obstacles_matrix[tuple(evaluated_position)] == -1:
            return obstacle_penalty

        # 计算价值得分，同时考虑到无人机位置到评估位置的距离
        value_score = signal_weight * signal_contribution - obstacle_weight * obstacle_contribution - distance_to_evaluated_position * distance_attenuation_factor
        return value_score


    def generate_local_value_matrix(self, local_signals_matrix, local_obstacles_matrix, drone_position):
        perception_range = int(self.config['perception_range'])
        matrix_size = 2 * perception_range + 1
        local_value_matrix = np.zeros((matrix_size, matrix_size))
        
        for i in range(matrix_size):
            for j in range(matrix_size):
                evaluated_position = np.array([i, j])
                local_value_matrix[i, j] = self.local_value_function(local_signals_matrix, local_obstacles_matrix, evaluated_position, drone_position)
        
        return local_value_matrix
        
    def generate_local_perception_data(self, drone_position):
        perception_range = int(self.config['perception_range'])
        matrix_size = 2 * perception_range + 1  # 计算局部矩阵的大小
        drone_position = np.array(drone_position, dtype=int)  # 确保drone_position是整数数组

        # 初始化局部矩阵
        local_signals_matrix = np.zeros((matrix_size, matrix_size))
        local_obstacles_matrix = np.zeros((matrix_size, matrix_size))

        # 计算局部矩阵的边界
        min_x = max(0, drone_position[0] - perception_range)
        max_x = min(self.config['environment_size'], drone_position[0] + perception_range + 1)
        min_y = max(0, drone_position[1] - perception_range)
        max_y = min(self.config['environment_size'], drone_position[1] + perception_range + 1)

        # 填充局部矩阵
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                local_x = x - drone_position[0] + perception_range
                local_y = y - drone_position[1] + perception_range
                if self.global_environment_matrix[x, y] == 1:
                    local_signals_matrix[local_x, local_y] = 1
                elif self.global_environment_matrix[x, y] == -1:
                    local_obstacles_matrix[local_x, local_y] = -1

        return {'signals': local_signals_matrix.tolist(), 'obstacles': local_obstacles_matrix.tolist()}


    def batch_generate_local_data(self):
        all_local_data = {}
        size = self.config['environment_size']
        for x in range(0, size, int(self.config['perception_range'])):
            for y in range(0, size, int(self.config['perception_range'])):
                drone_position = np.array([x, y], dtype=int)
                local_data = self.generate_local_perception_data(drone_position)
                local_value_matrix = self.generate_local_value_matrix(local_data['signals'], local_data['obstacles'], drone_position)
                all_local_data[f'position_{x}_{y}'] = {
                    'local_data': local_data,
                    'local_value_matrix': local_value_matrix.tolist()
                }
        return all_local_data


if __name__ == '__main__':
    config_path = 'config/config.yml'
    num_environments = 1024  # 生成4个环境
    output_dir = 'data/processed'  # 确保这个目录存在或代码中创建它

    # 实例化GlobalFlowDataPreparer并生成环境
    global_preparer = GlobalFlowDataPreparer(config_path)
    output_file = os.path.join(output_dir, 'all_global_data.json')
    global_preparer.batch_generate_environments(num_environments, output_file)

    # # 对于每个生成的环境，生成局部数据
    # all_local_data = {}
    # for i in range(num_environments):
    #     # 实例化LocalFlowDataPreparer并生成局部数据
    #     local_preparer = LocalFlowDataPreparer(output_file, config_path, i)
    #     local_data = local_preparer.batch_generate_local_data()
    #     all_local_data[f'environment_{i}'] = local_data
    #     print(f'Completed environment {i}.')

    # # 保存所有环境的局部数据到一个文件
    # all_local_data_file = os.path.join(output_dir, 'all_local_data.json')
    # with open(all_local_data_file, 'w') as f:
    #     json.dump(all_local_data, f, indent=4)
    # print(f'All local data has been saved to {all_local_data_file}.')
