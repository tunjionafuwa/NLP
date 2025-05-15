import pandas as pd

import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, dataset_file, day_range, split_date, train_test="train") -> None:
        df = pd.read_csv(dataset_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # standardize dataset
        df = (df - df.mean()) / df.std()

        self.mean = torch.tensor(df.mean().values).reshape(1, -1)
        self.std = torch.tensor(df.std().values).reshape(1, -1)

        if train_test == "train":
            self.dataset = df[df.index < split_date]
        elif train_test == "test":
            self.dataset = df[df.index >= split_date]
        else:
            raise ValueError("train_test must be either 'train' or 'test'")

        self.day_range = day_range

    def __getitem__(self, index):
        end_index = index + self.day_range
        current_series = self.dataset.iloc[index:end_index]

        day_tensor = torch.LongTensor(current_series.index.day.to_numpy())
        month_tensor = torch.LongTensor(current_series.index.month.to_numpy())
        data_values = torch.FloatTensor(current_series.values)

        return day_tensor, month_tensor, data_values

    def __len__(self):
        return len(self.dataset) - self.day_range
