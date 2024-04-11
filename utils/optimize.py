# 有待调整

import optuna
from src.main import SimulationManager

def objective(trial):
    # 定义超参数搜索空间
    dropout = trial.suggest_uniform('dropout', 0.1, 0.7)
    num_heads = trial.suggest_int('num_heads', 4, 16)
    sparsity = trial.suggest_uniform('sparsity', 0.05, 0.2)
    expansion = trial.suggest_int('expansion', 2, 6)
    negative_slope = trial.suggest_loguniform('negative_slope', 1e-3, 1e-1)
    eps = trial.suggest_loguniform('eps', 1e-6, 1e-4)

    # 更新配置文件
    config = {
        'models': {
            'mcts_vnet_model': {
                'dropout': dropout,
                'num_heads': num_heads,
                'sparsity': sparsity,
                'expansion': expansion,
                'negative_slope': negative_slope,
                'eps': eps,
                'shared_attention': True,
                'shared_norm_activation': False,
                'shared_positional_encoding': True
            }
        }
    }

    # 运行模拟并返回性能指标
    simulation_manager = SimulationManager(config)
    performance_metric = simulation_manager.simulate_interaction()
    return performance_metric

if __name__ == '__main__':
    # 创建一个Optuna研究对象并运行优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)  # 指定尝试的次数

    # 打印最佳超参数
    print('Best hyperparameters:', study.best_params)
