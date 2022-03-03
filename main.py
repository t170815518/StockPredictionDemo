# parse the argument
import argparse
import datetime
import logging
import random
import sys

import numpy as np
import torch
from DatasetLoader import DatasetLoader
from Trainer import Trainer
from Evaluator import Evaluator

parser = argparse.ArgumentParser(description='Meta-NER')
parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "linear_regression"])
parser.add_argument("--window", type=int, default=5)
parser.add_argument("--average_size", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--random_seed", type=int, default=1949)
args = parser.parse_args()

# set random seed
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# setup logger
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("train.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# log the arguments
logging.info(args)

data = DatasetLoader(data_path="MT.csv", window_size=args.window, average_size=args.average_size)
evaluator = Evaluator(data)

if args.average_size <= 0:
    feature_num = 4
else:
    feature_num = 8
    logging.info("Moving average is used")

trainer = Trainer(model_type=args.model.lower(), input_size=feature_num, data_loader=data, learning_rate=args.lr,
                  evaluator=evaluator)

# start training
trainer.train()
