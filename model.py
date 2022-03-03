#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn


class Model:
    def __init__(self, is_add_moving_average: bool, window_size: int):
        self.window_size = window_size
        self.is_add_moving_average = is_add_moving_average

    def predict(self, x):
        pass


class LSTM(torch.nn.Module):
    """
    Uni-directional LSTM
    Ref: https://www.kaggle.com/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch#3.-Build-the-structure
    -of-model
    """

    def __init__(self, hidden_size: int, input_size: int, device):
        super(LSTM, self).__init__()

        self.device = device

        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0, num_layers=1)
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2, bias=True)
        self.relu = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4, bias=True)
        self.relu2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_size // 4, 1, bias=True)

    def forward(self, X):
        # convert numpy array to tensor
        X = torch.Tensor(X).to(self.device)

        out, (last_hidden, _) = self.lstm_layer(X)
        y = self.linear1(out[:, -1, :])
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.linear3(y)
        return y
