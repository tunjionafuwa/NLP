import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.Dataset import WeatherDataset


# Max Daily Temperature and Rainfall Dataset
dataset_file = "./data/weather.csv"
split_date = pd.to_datetime("2023-01-01")
day_range = 15
days_in = 14

assert day_range > days_in, "day_range must be greater than days_in"

learning_rate = 1e-4
num_epochs = 500
batch_size = 32

dataset_train = WeatherDataset(dataset_file, day_range, split_date, "train")
dataset_test = WeatherDataset(dataset_file, day_range, split_date, "test")

data_loader_train = DataLoader(
    dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
)
data_loader_test = DataLoader(
    dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True
)

fig = plt.figure(figsize=(10, 5))
plt.plot(dataset_train.dataset.index, dataset_train.dataset.values[:, 1], label="Train")
plt.plot(dataset_test.dataset.index, dataset_test.dataset.values[:, 1], label="Test")
plt.title("Max Daily Temperature and Rainfall")
plt.xlabel("Date")
plt.legend()
plt.show()


class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.norm1 = nn.LayerNorm(input_size // 2)

        self.fc2 = nn.Linear(input_size // 2, output_size)

        self.fc3_skip = nn.Linear(input_size, output_size)
        self.norm_skip = nn.LayerNorm(input_size)

        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.norm_skip(x))
        skip = self.fc3_skip(x)

        x = self.elu(self.norm1(self.fc1(x)))
        x = self.fc2(x)

        output = x + skip
        return output


class ResMLP(nn.Module):
    def __init__(self, seq_len, output_size, num_blocks=1):
        super(ResMLP, self).__init__()

        seq_data_len = seq_len * 2
        self.input_mlp = nn.Sequential(
            nn.Linear(seq_data_len, 4 * seq_data_len),
            nn.ELU(),
            nn.LayerNorm(4 * seq_data_len),
            nn.Linear(4 * seq_data_len, 128),
        )

        blocks = [ResBlockMLP(128, 128) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        self.fc_out = nn.Linear(128, output_size)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.input_mlp(x)

        x = self.elu(self.res_blocks(x))
        return self.fc_out(x)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

weather_mlp = ResMLP(seq_len=days_in, output_size=2).to(device)
optimizer = optim.Adam(weather_mlp.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

num_model_params = 0
for param in weather_mlp.parameters():
    num_model_params += param.flatten().shape[0]


training_loss_logger = []
for epoch in range(num_epochs):
    weather_mlp.train()
    for i, (day, month, data_seq) in enumerate(data_loader_train):
        seq_block = data_seq[:, :days_in].to(device)
        loss = 0
        for i in range(day_range - days_in):
            target_seq_block = data_seq[:, i + days_in].to(device)
            data_pred = weather_mlp(seq_block)

            loss += criterion(data_pred, target_seq_block)
            seq_block = torch.cat(
                (seq_block[:, 1:, :], data_pred.unsqueeze(dim=1)), dim=1
            ).detach()

        loss /= i + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss_logger.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

plt.figure(figsize=(10, 5))
plt.plot(training_loss_logger)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


data_tensor = torch.FloatTensor(dataset_test.dataset.values)
log_predictions = []
weather_mlp.eval()
with torch.no_grad():
    seq_block = data_tensor[:days_in, :].unsqueeze(0).to(device)
    for i in range(data_tensor.shape[0] - days_in):
        data_pred = weather_mlp(seq_block)
        log_predictions.append(data_pred.cpu())

        seq_block = torch.cat(
            (seq_block[:, 1:, :], data_pred.unsqueeze(dim=1)), dim=1
        ).detach()

predictions_cat = torch.cat(log_predictions)
un_norm_predictions = predictions_cat * dataset_test.std + dataset_test.mean
un_norm_data = data_tensor * dataset_test.std + dataset_test.mean
un_norm_data = un_norm_data[days_in:]

test_mse = (un_norm_data - un_norm_predictions).pow(2).mean().item()
print(f"Test MSE: {test_mse}")

plt.figure(figsize=(10, 5))
plt.plot(un_norm_data[:, 0], label="Ground Truth")
plt.plot(un_norm_predictions[:, 0], label="Predicted")
plt.title("Rainfall in (mm)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(un_norm_data[:, 1], label="Ground Truth")
plt.plot(un_norm_predictions[:, 1], label="Predicted")
plt.title("Max Daily Temperature (C)")
plt.legend()
plt.show()
