# -*- coding: utf-8 -*-
import os
import json
import errno
from tensorflow.keras.utils import plot_model

# save model architecture as .png file
def plot_model_architecture(model, filepath="model.png"):
    plot_model(model, to_file=filepath, dpi=600)

# create directory (if it does not exist)
def create_dir_if_not_exists(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# initialize experiment directory and sub-directories
def initialize_experiment_directory(experiment_name):
    create_dir_if_not_exists(experiment_name)
    create_dir_if_not_exists(experiment_name + "/data")

# write hyperparameters to directory (and to std output)
def write_hyperparameters(argparse_args, dir_path, verbose=False):
    hyperparameters = {
        "experiment_name": argparse_args.experiment_name,
        "num_rows": argparse_args.num_rows,
        "window_size": argparse_args.window_size,
        "embedding_dim": argparse_args.embedding_dim,
        "epochs": argparse_args.epochs,
        "learning_rate": argparse_args.learning_rate,
        "random_seed": argparse_args.random_seed
    }
    json.dump(hyperparameters, open(dir_path + "/hyperparameters.json", "w"), indent=4)
    if verbose:
        print("")
        for (k, v) in hyperparameters.items():
            print(" - {arg_name}: {value}".format(arg_name=k, value=v))
        print("")