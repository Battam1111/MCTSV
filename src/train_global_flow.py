import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.utils.data import random_split

import sys

# 将项目根目录添加到 sys.path 中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.global_flow_model import GlobalFlowModel
from datasets.global_flow_dataset import GlobalFlowDataset
import yaml
import wandb


def load_config(config_path="config/config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, save_dir, val_loss, filename_prefix="model"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f"{filename_prefix}_val_loss{val_loss:.4f}.pt"
    save_path = os.path.join(save_dir, filename)
    for file in os.listdir(save_dir):
        if file.startswith(filename_prefix) and file.endswith('.pt'):
            os.remove(os.path.join(save_dir, file))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train_model(model, train_dataset, val_dataset, config):
    wandb.init(project="global_flow_project", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 10
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch}: Train Loss {train_loss}, Validation Loss {val_loss}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "saved_models/global_flow", val_loss, "global_flow_model")
            print(f"Saved model with Validation Loss {val_loss}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max_epochs_without_improvement:
                print("Early stopping triggered")
                break
    wandb.finish()

def main():
    config = load_config()
    dataset = GlobalFlowDataset(config['datasets']['global_flow']['data_file'], config['models']['global_flow_model'])
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    model = GlobalFlowModel(config['models']['global_flow_model'])
    train_model(model, train_dataset, val_dataset, config)

if __name__ == "__main__":
    main()
