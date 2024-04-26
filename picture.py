import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 初始化wandb
wandb.login()

# 定义项目和实体
project_name = 'MCTSV-online_learning'
entity = 'battam'

# 设置绘图风格和颜色
sns.set(style="whitegrid")
palette = sns.color_palette("viridis", n_colors=5)

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 初始化存储结果的DataFrame
results_df = pd.DataFrame(columns=['run_name', '平均数据收集率', '平均能效', 'average_reward'])

# 从wandb获取数据
api = wandb.Api()
runs = api.runs(f"{entity}/{project_name}")

for run in runs:
    if run.name in ["在线学习+MCTS", "无训练-MCTS", "在线学习", "无训练", "随机"]:
        data = run.history(samples=5000)
        data['data_collection_rate'] = data['collected_points'] / data['max_num_signal_points']
        data['energy_efficiency'] = data['data_collection_rate'] / 1
        average_reward = data['reward'].sum() / len(data['reward'])
        
        temp_df = pd.DataFrame({
            'run_name': [run.name],
            '平均数据收集率': [data['data_collection_rate'].mean()],
            '平均能效': [data['energy_efficiency'].mean()],
            'average_reward': [average_reward]
        })
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

def plot_bar_chart(x, y, data, title, filename):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=x, y=y, hue=x, data=data, palette=palette, dodge=False)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel('')
    plt.ylabel(y, fontsize=14)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_bar_chart('run_name', '平均数据收集率', results_df, '平均数据收集率', 'data_collection_rate.png')
plot_bar_chart('run_name', '平均能效', results_df, '平均能效', 'energy_efficiency.png')
plot_bar_chart('run_name', 'average_reward', results_df, '平均奖励', 'average_reward.png')
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 初始化wandb
wandb.login()

# 定义项目和实体
project_name = 'MCTSV-online_learning'
entity = 'battam'

# 设置绘图风格和颜色
sns.set(style="whitegrid")
palette = sns.color_palette("viridis", n_colors=5)

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 初始化存储结果的DataFrame
results_df = pd.DataFrame(columns=['run_name', '平均数据收集率', '平均能效', 'average_reward'])

# 从wandb获取数据
api = wandb.Api()
runs = api.runs(f"{entity}/{project_name}")

for run in runs:
    if run.name in ["在线学习+MCTS", "无训练-MCTS", "在线学习", "无训练", "随机"]:
        data = run.history(samples=5000)
        data['data_collection_rate'] = data['collected_points'] / data['max_num_signal_points']
        data['energy_efficiency'] = data['data_collection_rate'] / 1
        average_reward = data['reward'].sum() / len(data['reward'])
        
        temp_df = pd.DataFrame({
            'run_name': [run.name],
            '平均数据收集率': [data['data_collection_rate'].mean()],
            '平均能效': [data['energy_efficiency'].mean()],
            'average_reward': [average_reward]
        })
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

def plot_bar_chart(x, y, data, title, filename):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=x, y=y, hue=x, data=data, palette=palette, dodge=False)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel('')
    plt.ylabel(y, fontsize=14)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_bar_chart('run_name', '平均数据收集率', results_df, '平均数据收集率', 'data_collection_rate.png')
plot_bar_chart('run_name', '平均能效', results_df, '平均能效', 'energy_efficiency.png')
plot_bar_chart('run_name', 'average_reward', results_df, '平均奖励', 'average_reward.png')
