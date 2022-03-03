#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

from DatasetLoader import DatasetLoader


class Evaluator:
    def __init__(self, data_loader: DatasetLoader):
        self.device = None

        self.data_loader = data_loader
        self.loss_func = torch.nn.MSELoss()

    def plot_trend(self):
        """plot the actual and predicted stock trending"""
        pass

    def evaluate(self, model):
        """

        :param model:
        :return:
        """
        total_loss = 0
        batch_counter = 0

        with torch.no_grad():
            for X, y in self.data_loader.iter_test():
                pred_y = model(X)
                y = torch.FloatTensor(y).to(self.device)
                loss = self.loss_func(pred_y.squeeze(1), y)
                total_loss += loss.item()
                batch_counter += 1

        eval_loss = total_loss / batch_counter
        return eval_loss

    def final_evaluate(self, model, is_neural_model: bool = True):
        if is_neural_model:
            with torch.no_grad():
                predicted_values = []
                actual_values = []

                for X, y in self.data_loader.iter_train(is_shuffle=False):
                    pred_y = model(X)
                    pred_y = pred_y.detach().cpu().squeeze(1).numpy()
                    predicted_values.append(pred_y)
                    actual_values.append(y)

                predicted_values = np.concatenate(predicted_values)
                actual_values = np.concatenate(actual_values)

                # create result pandas
                results = pd.DataFrame({"actual values": actual_values, "predicted values": predicted_values})
                results.to_csv("train_result.csv")

                predicted_values = []
                actual_values = []

                for X, y in self.data_loader.iter_test():
                    pred_y = model(X)
                    pred_y = pred_y.detach().cpu().squeeze(1).numpy()
                    predicted_values.append(pred_y)
                    actual_values.append(y)

                predicted_values = np.concatenate(predicted_values)
                actual_values = np.concatenate(actual_values)

                # create result pandas
                results = pd.DataFrame({"actual values": actual_values, "predicted values": predicted_values})
                results.to_csv("test_result.csv")
        else:
            train_data = np.copy(self.data_loader.train)
            X = train_data[:, :-1, :]
            y = train_data[:, -1, 0]
            sample_num = X.shape[0]
            X = X.reshape(sample_num, -1)
            pred_y = model.predict(X)
            results = pd.DataFrame({"actual values": y, "predicted values": pred_y})
            results.to_csv("train_result.csv")
            error = mean_squared_error(y, pred_y)
            logging.info("train error (MSE) = {}".format(error))

            test_data = np.copy(self.data_loader.test)
            X = test_data[:, :-1, :]
            y = test_data[:, -1, 0]
            sample_num = X.shape[0]
            X = X.reshape(sample_num, -1)
            pred_y = model.predict(X)
            results = pd.DataFrame({"actual values": y, "predicted values": pred_y})
            results.to_csv("test_result.csv")

            error = mean_squared_error(y, pred_y)
            logging.info("test error (MSE) = {}".format(error))