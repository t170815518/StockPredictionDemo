#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging

import torch
import numpy as np
from torch.optim import SGD, Adam
from sklearn.linear_model import LinearRegression

from DatasetLoader import DatasetLoader
from Evaluator import Evaluator
from model import LSTM


class Trainer:
    def __init__(self, model_type: str, input_size: int, data_loader: DatasetLoader, learning_rate: float,
                 evaluator: Evaluator, verbose_interval: int = 1,
                 patience_num: int = 10, max_iteration: int = 999,
                 evaluate_interval: int = 1):
        self.evaluator = evaluator
        self.evaluate_interval = evaluate_interval
        self.patience_num = patience_num
        self.verbose_interval = verbose_interval
        self.data_loader = data_loader
        self.max_iteration = max_iteration

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("device = {}".format(self.device))
        self.evaluator.device = self.device

        # construct  the model to train
        self.model = None
        self.is_neural_network = False
        if model_type == "lstm":
            self.model = LSTM(120, input_size, device=self.device)
            self.is_neural_network = True
            self.model.to(self.device)
            self.loss_func = torch.nn.MSELoss()
            self.optimizer = Adam(self.model.parameters(), learning_rate)
        elif model_type == "linear_regression":
            self.model = LinearRegression(n_jobs=-1)
        else:
            raise ValueError("{} is an unknown model".format(model_type))

    def train(self):
        """
        Train with a fixed set of parameter
        :return:
        """
        if self.is_neural_network:
            self.train_neural_network()
        else:  # normal linear model
            train_data = np.copy(self.data_loader.train)
            X = train_data[:, :-1, :]
            y = train_data[:, -1, 0]
            sample_num = X.shape[0]
            X = X.reshape(sample_num, -1)
            self.model.fit(X, y)
            self.evaluator.final_evaluate(self.model, is_neural_model=False)

    def train_neural_network(self):
        patience_counter = 0
        min_loss = float('inf')
        for i in range(self.max_iteration):
            iter_id = 0
            epoch_total_loss = 0
            # train epoch
            for X, y in self.data_loader.iter_train():
                iter_id += 1
                pred_y = self.model.forward(X)
                y = torch.FloatTensor(y).to(self.device)
                loss = self.loss_func(pred_y.squeeze(1), y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_total_loss += loss.item()

                if iter_id % self.verbose_interval == 0:
                    # verbose the average loss
                    logging.info("[Epoch{} Iteration{}] average_loss = {}".format(i, iter_id,
                                                                                  epoch_total_loss / iter_id))

            if (i + 1) % self.evaluate_interval == 0:
                loss = self.evaluator.evaluate(self.model)
                if loss < min_loss:
                    min_loss = loss
                    patience_counter = 0
                    logging.info("Minimum loss found = {}".format(min_loss))
                else:
                    patience_counter += 1
                if patience_counter >= self.patience_num:
                    break  # early breaking
        self.evaluator.final_evaluate(self.model)
        logging.info("Training completes")
