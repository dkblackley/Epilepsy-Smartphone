# -*- coding: utf-8 -*-


"""
This file contains the following functions to read arguments from a configurations file and command line arguments.
    load_arguments - Function to load arguments from a configurations file and command line arguments.
    print_arguments - Function to print the name and value for each loaded argument.
"""


# Built-in/Generic Imports
import sys
from configparser import ConfigParser
from argparse import ArgumentParser, Namespace

# Own Module Imports
from utils import log, string_to_boolean


__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy analysis for smartphone video"
__credits__    = ["Daniel Blackley", "Emannuele Trucco", "Stephen McKenna"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


def load_arguments(description: str) -> Namespace:
    """
    Loads arguments from a config file and command line.
    Arguments from command line overrides arguments from the config file.
    The config file will be loaded from the default location ./config.ini and can be overridden from the command line.
    :param description: The description of the application.
    :return: ArgumnetParser Namespace containing loaded arguments.
    """

    # Creates an ArgumentParser to read command line arguments.
    argument_parser = ArgumentParser(description=description)

    # Creates a ConfigParser to read the config file.
    config_parser = ConfigParser()

    # Loads either a specified config file or default config file.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config_file":
            config_parser.read(sys.argv[2])
        else:
            config_parser.read("config.ini")
    else:
        config_parser.read("config.ini")

    # Standard Arguments
    argument_parser.add_argument("--config_file", type=str,
                                 default="config.ini",
                                 help="String representing the file path to the config file.")
    argument_parser.add_argument("--seed", type=int,
                                 default=int(config_parser["standard"]["seed"]),
                                 help="Integer for the seed used to generate random numbers.")
    argument_parser.add_argument("--task", type=str,
                                 default=config_parser["standard"]["task"].lower(),
                                 help="String for the task to be performed.")
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String for the name of the current experiment.")

    # Logging Arguments
    argument_parser.add_argument("--verbose", type=string_to_boolean,
                                 default=string_to_boolean(config_parser["logging"]["verbose"]),
                                 help="Boolean value if outputs should be printed to the console.")
    argument_parser.add_argument("--log_dir", type=str,
                                 default=config_parser["logging"]["log_dir"],
                                 help="Directory path for where the logs will be saved.")
    argument_parser.add_argument("--model_dir", type=str,
                                 default=config_parser["logging"]["model_dir"],
                                 help="Directory path for where the trained models will be saved.")
    argument_parser.add_argument("--tensorboard_dir", type=str,
                                 default=config_parser["logging"]["tensorboard_dir"],
                                 help="Directory path for where the TensorBoard logs will be saved.")

    # Dataset Arguments
    argument_parser.add_argument("--dataset_dir", type=str,
                                 default=config_parser["dataset"]["dataset_dir"],
                                 help="Directory path for the location of the dataset.")
    argument_parser.add_argument("--augmentation", type=string_to_boolean,
                                 default=string_to_boolean(config_parser["dataset"]["augmentation"]),
                                 help="Boolean if augmentation should be used to load the dataset.")
    argument_parser.add_argument("--validation_split", type=float,
                                 default=string_to_boolean(config_parser["dataset"]["augmentation"]),
                                 help="Floating point value for the percentage of data used for validation.")
    argument_parser.add_argument("--image_x", type=int,
                                 default=int(config_parser["dataset"]["image_x"]),
                                 help="Integer for the x dimension of the image after resizing.")
    argument_parser.add_argument("--image_y", type=int,
                                 default=int(config_parser["dataset"]["image_y"]),
                                 help="Integer for the y dimension of the image after resizing.")

    # Performance Arguments
    argument_parser.add_argument("--efficient_net", type=int,
                                 default=int(config_parser["performance"]["efficient_net"]),
                                 help="Integer for the compound coefficient for the EfficientNet model.")
    argument_parser.add_argument("--float16", type=string_to_boolean,
                                 default=string_to_boolean(config_parser["performance"]["float16"]),
                                 help="Boolean if 16 bit precision should be used.")
    argument_parser.add_argument("--cuda", type=string_to_boolean,
                                 default=string_to_boolean(config_parser["performance"]["cuda"]),
                                 help="Boolean if a CUDA device should be used if available.")
    argument_parser.add_argument("--data_workers", type=int,
                                 default=int(config_parser["performance"]["data_workers"]),
                                 help="Integer for the number of data workers used for processing the dataset.")

    # Training Arguments
    argument_parser.add_argument("--epochs", type=int,
                                 default=int(config_parser["training"]["epochs"]),
                                 help="Integer for the number of training epochs to perform.")
    argument_parser.add_argument("--batch_size", type=int,
                                 default=int(config_parser["training"]["batch_size"]),
                                 help="Integer for the size of the batches used to train the model.")
    argument_parser.add_argument("--learning_rate", type=float,
                                 default=float(config_parser["training"]["learning_rate"]),
                                 help="Floating point value for the learning rate used to train the model.")
    argument_parser.add_argument("--pretraining", type=string_to_boolean,
                                 default=string_to_boolean(config_parser["training"]["pretraining"]),
                                 help="Boolean if the model should be initialised with pretrained weights.")

    # Monte Carlo Dropout Arguments
    argument_parser.add_argument("--drop_chance", type=float,
                                 default=float(config_parser["mc_dropout"]["drop_chance"]),
                                 help="Floating point value for the drop probability of dropout layers.")
    argument_parser.add_argument("--drop_iterations", type=int,
                                 default=int(config_parser["mc_dropout"]["drop_iterations"]),
                                 help="Integer for the number of Monte Carlo iterations to be performed.")

    # Bayes by Backprop Arguments
    argument_parser.add_argument("--bbb_iterations", type=int,
                                 default=int(config_parser["bayes_by_backprop"]["bbb_iterations"]),
                                 help="Integer for the number of Bayes by Backprop iterations to be performed.")

    # Debug Arguments
    argument_parser.add_argument("--batches_per_epoch", type=int,
                                 default=int(config_parser["debug"]["batches_per_epoch"]),
                                 help="Integer for the number of batches to perform per epoch. 0 for all.")

    # Returns the argument parser.
    return argument_parser.parse_args()


def print_arguments(arguments: Namespace) -> None:
    """
    Logs all input arguments.
    :param arguments: Namespace containing all arguments.
    """

    for key, value in vars(arguments).items():
        log(arguments, f"{key: <24}: {value}")