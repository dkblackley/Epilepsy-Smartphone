# -*- coding: utf-8 -*-


"""
This file contains the following utility functions for the application:
    string_to_boolean - Function to convert String to a Boolean value.
    log - Function to both print and log a given input message.
    set_random_seed - Function that sets a seed for all random number generation functions.
"""


# Built-in/Generic Imports
import os
import random
from argparse import ArgumentTypeError, Namespace

# Library Imports
import torch
import numpy as np


__author__     = ["Daniel Blackley", "Jacob Carse"]
__copyright__  = "Copyright 2021, Cost-Sensitive Selective Classification for Skin Lesions"
__credits__    = ["Daniel Blackley", "Jacob Carse", "Stephen McKenna"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@dundee.ac.uk"
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


def log(arguments: Namespace, message: str) -> None:
    """
    Logging function that will both print and log an input message.
    :param arguments: An ArgumentParser Namespace containing loaded arguments.
    :param message: String containing message to be printed and logged.
    """

    # Prints the input message is running in verbose mode.
    if arguments.verbose:
        print(message)

    # Logs a message to a specified log directory if specified.
    if arguments.log_dir != '':

        # Creates a folder if file path does not exist.
        os.makedirs(os.path.dirname(arguments.log_dir), exist_ok=True)

        # Appends the message to the log file.
        print(message, file=open(os.path.join(arguments.log_dir, f"{arguments.experiment}_log.txt"), 'a'))


def set_random_seed(seed: int) -> None:
    """
    Sets a random seed for each python library that generates random numbers.
    :param seed: Integer for the number used as the seed.
    """

    # Sets the seed for Python in built functions.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Sets the random seed for the Python libraries.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sets PyTorch to be deterministic if using CUDNN.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cuda: bool = True) -> torch.device:
    """
    Sets the device that will be used for training and testing.
    :param cuda: Boolean value for if a cuda device should be used.
    :return: A PyTorch device.
    """

    # Checks if a CUDA device is available to be used.
    if cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.device_count() - 1}")

    # Sets the device to CPU.
    else:
        return torch.device("cpu")