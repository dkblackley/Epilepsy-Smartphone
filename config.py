# -*- coding: utf-8 -*-


"""
This file contains the following functions to read arguments from a configurations file and command line arguments.
    load_arguments - Function to load arguments from a configurations file and command line arguments.
    print_arguments - Function to print the name and value for each loaded argument.
"""


# Built-in/Generic Imports
import sys
from configparser import ConfigParser
from argparse import ArgumentParser, Namespace, ArgumentTypeError


__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy-Smartphone"
__credits__    = ["Daniel Blackley", "Stephen McKenna", "Emanuele Trucco"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


def string_to_boolean(argument: str) -> bool:
    """
    Converts a String to a Boolean value.
    :param argument: Input String to be converted.
    :return: Boolean value.
    """

    # Checks if the input value is already a Boolean.
    if isinstance(argument, bool):
        return argument

    # Checks if the String value is equal to true.
    elif argument.lower() == "true" or argument.lower() == 't':
        return True

    # Checks if the String value is equal to false.
    elif argument.lower() == "false" or argument.lower() == 'f':
        return False

    # Returns an Argument Type Error if no boolean value was found.
    else:
        raise ArgumentTypeError("Boolean value expected.")


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
    argument_parser.add_argument("--debug", type=string_to_boolean,
                                 default=config_parser["standard"]["debug"],
                                 help="Directory path for where the logs will be saved.")
    argument_parser.add_argument("--model_dir", type=str,
                                 default=config_parser["standard"]["model_dir"],
                                 help="Directory path for where the trained models will be saved.")
    argument_parser.add_argument("--cuda", type=string_to_boolean,
                                 default=string_to_boolean(config_parser["standard"]["cuda"]),
                                 help="Boolean if a CUDA device should be used if available.")


    # Training Arguments
    argument_parser.add_argument("--epochs", type=int,
                                 default=int(config_parser["training"]["epochs"]),
                                 help="Integer for the number of training epochs to perform.")
    argument_parser.add_argument("--learning_rate", type=float,
                                 default=float(config_parser["training"]["learning_rate"]),
                                 help="Float to decide the learning rate of the Adam optimiser.")
    argument_parser.add_argument("--dataset_dir", type=str,
                                 default=config_parser["training"]["dataset_dir"],
                                 help="Directory path for the location of the dataset.")
    argument_parser.add_argument("--validation_split", type=float,
                                 default=string_to_boolean(config_parser["training"]["val_split"]),
                                 help="Floating point value for the percentage of data used for validation.")
    argument_parser.add_argument("--segment", type=float,
                                 default=str(config_parser["training"]["segment"]),
                                 help="The ROI to segment")

    # Returns the argument parser.
    return argument_parser.parse_args()
