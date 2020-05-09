# -*- coding: utf-8 -*-
import argparse
from clean import clean_raw_data
from preprocess import preprocess_data
from train import build_and_train_model
from utils import initialize_experiment_directory, write_hyperparameters

# build argument parser
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", help="name of experiment")
parser.add_argument("csv_path", help="path to original .csv dataset")
parser.add_argument("-nr", "--num_rows", type=int, help="number of rows to extract (-1 for whole dataset)", default=10000)
parser.add_argument("-wsz", "--window_size", type=int, help="(half) window size for skip-gram contexts", default=2)
parser.add_argument("-emb", "--embedding_dim", type=int, help="word2vec embedding dimension size", default=100)
parser.add_argument("-e", "--epochs", type=int, help="number of training epochs", default=20)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate for Adam optimizer", default=0.001)
parser.add_argument("-rs", "--random_seed", type=int, help="random seed for skip-gram generation and training", default=42)
args = parser.parse_args()

# initialize experiment directory
initialize_experiment_directory(args.experiment_name)
write_hyperparameters(args, args.experiment_name, verbose=True)

# clean raw data
clean_raw_data(args.experiment_name, args.csv_path, args.num_rows)

# preprocess dataset
preprocess_data(args.experiment_name, args.window_size, args.random_seed)

# train data
build_and_train_model(args.experiment_name, args.embedding_dim, args.learning_rate, args.epochs, args.random_seed)