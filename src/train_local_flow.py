import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from models.local_flow_model import LocalFlowTransformer
from datasets.local_flow_dataset import LocalFlowDataset
import yaml
import wandb
import os
from torch.optim.lr_scheduler import StepLR

def load_config(config_path="config/config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, save_dir, val_loss, filename_prefix="model"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f"{filename_prefix}_val_loss{val_loss:.4f}.pt"
    save_path = os.path.join(save_dir, filename)

    # 删除之前的最佳模型文件
    for file in os.listdir(save_dir):
        if file.startswith(filename_prefix) and file.endswith('.pt'):
            os.remove(os.path.join(save_dir, file))

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train_model(model, train_dataset, val_dataset, config):
    wandb.init(project="local_flow_project", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])  # 更改为AdamW
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 添加学习率调度器
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            signals, obstacles, value_matrix = batch['signals'].to(device), batch['obstacles'].to(device), batch['value_matrix'].to(device)
            optimizer.zero_grad()
            output = model(signals)
            loss = criterion(output, value_matrix)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                signals, obstacles, value_matrix = batch['signals'].to(device), batch['obstacles'].to(device), batch['value_matrix'].to(device)
                output = model(signals)
                loss = criterion(output, value_matrix)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        print(f"Epoch {epoch}: Train Loss {train_loss}, Validation Loss {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "saved_models/local_flow", val_loss, "local_flow_model")
            print(f"Saved model with Validation Loss {val_loss}")

        scheduler.step()  # 更新学习率

    wandb.finish()

def main():
    config = load_config()
    # print(config['local_flow_dataset']['data_dir'])
    file_path = config['local_flow_dataset']['data_dir']

    dataset = LocalFlowDataset(file_path)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    print("Train size:", train_size, "Val size:", val_size)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = LocalFlowTransformer(config['models']['local_flow_model'])

    train_model(model, train_dataset, val_dataset, config)

if __name__ == "__main__":
    main()
