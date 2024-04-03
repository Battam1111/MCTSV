import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib
# matplotlib.use('TkAgg')  # 使用TkAgg后端

class SimulationVisualizer:
    def __init__(self, environment, update_interval=1000):
        plt.ion()
        self.environment = environment  # 保存环境实例的引用
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.colors = {0: '#f0f0f0', -1: '#303030', 1: '#76b947'}
        self.update_interval = update_interval
        self.init_ax()

    def init_ax(self):
        self.ax.clear()
        self.ax.set_xlim(-1, 17)
        self.ax.set_ylim(-3, 11)
        plt.axis('off')

    def draw_simulation(self, global_matrix, local_matrix, drone_position, battery, reward):
        self.init_ax()  # Re-initialize the axes for the new frame
        ax = self.ax

        # 绘制全局矩阵
        for i in range(global_matrix.shape[0]):
            for j in range(global_matrix.shape[1]):
                color = self.colors[global_matrix[i, j]]
                rect = patches.Rectangle((j, 10-i), 1, 1, linewidth=0.5, edgecolor='black', facecolor=color)
                ax.add_patch(rect)

        # 标记无人机位置
        drone = patches.Rectangle((drone_position[1], 10-drone_position[0]), 1, 1, linewidth=0.5, edgecolor='black', facecolor='#4287f5')
        ax.add_patch(drone)

        # 设置局部矩阵位置和大小
        local_matrix_offset = 11
        for i in range(local_matrix.shape[0]):
            for j in range(local_matrix.shape[1]):
                color = self.colors[local_matrix[i, j]]
                rect = patches.Rectangle((j+local_matrix_offset, 5-i), 1, 1, linewidth=0.5, edgecolor='black', facecolor=color)
                ax.add_patch(rect)

        # 添加电池电量和奖励信息
        plt.text(0, -1, f'Battery: {battery}', fontsize=10, family='sans-serif')
        plt.text(0, -2, f'Reward: {reward}, Task Completion: Not Done', fontsize=10, family='sans-serif')

    def start_animation(self):
        
        def update(frame):
            # 获取环境的当前状态
            state = self.environment.state
            global_matrix = state['global_matrix']
            local_matrix = state['local_matrix']
            drone_position = state['drone_position']
            battery = state['battery']
            reward = self.environment.reward

            self.draw_simulation(global_matrix, local_matrix, drone_position, battery, reward)
        
        ani = FuncAnimation(self.fig, update, frames=np.arange(0, 100), init_func=self.init_ax, blit=False, repeat=True, interval=self.update_interval, save_count=100)
        plt.draw()  # 更新图像
        plt.pause(1)  # 短暂暂停以更新图像并允许其他处理

# Example of how to use this class
if __name__ == "__main__":
    visualizer = SimulationVisualizer()
    visualizer.start_animation()
