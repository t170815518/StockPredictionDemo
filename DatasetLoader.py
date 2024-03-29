#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from numpy.random import shuffle
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


class DatasetLoader:
    def __init__(self, data_path: str, window_size: int, average_size: int = 0, split_ratio: float = 0.7,
                 batch_size: int = 32):
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.data_path = data_path

        self.train = None
        self.test = None

        # read data
        complete_data = pd.read_csv(data_path)
        # reverse the data
        complete_data = complete_data.iloc[::-1]
        # split for train and test
        train_end = int(len(complete_data) * (1 - split_ratio))
        train_data = complete_data.loc[:train_end]
        test_data = complete_data.loc[train_end:]

        if average_size > 0:
            train_data["move_average_close"] = train_data["close"].rolling(average_size, closed="left").mean()
            train_data["move_average_high"] = train_data["high"].rolling(average_size, closed="left").mean()
            train_data["move_average_low"] = train_data["low"].rolling(average_size, closed="left").mean()
            train_data["move_average_open"] = train_data["open"].rolling(average_size, closed="left").mean()

            test_data["move_average_close"] = test_data["close"].rolling(average_size, closed="left").mean()
            test_data["move_average_high"] = test_data["high"].rolling(average_size, closed="left").mean()
            test_data["move_average_low"] = test_data["low"].rolling(average_size, closed="left").mean()
            test_data["move_average_open"] = test_data["open"].rolling(average_size, closed="left").mean()

            train_data = train_data.iloc[average_size:]
            test_data = test_data.iloc[average_size:]

            train_data = train_data[["close", "high", "low", "open", "move_average_close", "move_average_high",
                                     "move_average_low", "move_average_open"]].to_numpy()
            test_data = test_data[["close", "high", "low", "open", "move_average_close", "move_average_high",
                                   "move_average_low", "move_average_open"]].to_numpy()
        else:
            # select only relevant columns
            train_data = train_data[["close", "high", "low", "open"]].to_numpy()
            test_data = test_data[["close", "high", "low", "open"]].to_numpy()

        # preprocess the data
        # normalize the data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        # get the sliding window
        if average_size > 0:
            train_data = sliding_window_view(train_data, window_shape=(window_size + 1, 8)).squeeze()
            test_data = sliding_window_view(test_data, window_shape=(window_size + 1, 8)).squeeze()
        else:
            train_data = sliding_window_view(train_data, window_shape=(window_size + 1, 4)).squeeze()
            test_data = sliding_window_view(test_data, window_shape=(window_size + 1, 4)).squeeze()

        # store the processed values
        self.train = train_data
        self.test = test_data

    def iter_train(self, is_shuffle=True):
        train_data = np.copy(self.train)
        if is_shuffle:
            shuffle(train_data)

        for start_id in range(0, len(train_data), self.batch_size):
            end_id = min(len(train_data), start_id + self.batch_size)
            batch = train_data[start_id: end_id, :]

            # process the batch
            X = batch[:, :-1, :]
            y = batch[:, -1, 0]  # y: the close price to predict

            yield X, y

    def iter_test(self):
        for start_id in range(0, len(self.test), self.batch_size):
            end_id = min(len(self.test), start_id + self.batch_size)
            batch = self.test[start_id: end_id, :]

            # process the batch
            X = batch[:, :-1, :]
            y = batch[:, -1, 0]  # y: the close price to predict

            yield X, y


if __name__ == '__main__':
    for X, y in DatasetLoader("600519.csv", 3).iter_train():
        print(X, y)