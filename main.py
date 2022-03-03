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
parser.add_argument("--model", type=str, default="lstm")
parser.add_argument("--window", type=int, default=50)
parser.add_argument("--average_move", action="store_true")
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

data = DatasetLoader(data_path="600519.csv", window_size=args.window)
evaluator = Evaluator(data)
trainer = Trainer(model_type=args.model.lower(), input_size=4, is_add_moving_average=args.average_move,
                  data_loader=data, learning_rate=args.lr, evaluator=evaluator)

# start training
trainer.train()
