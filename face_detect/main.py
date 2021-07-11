#! /usr/bin/python
# -*- coding: utf-8 -*-


"""
This file is the main executable for Cost-Sensitive Selective Classification for Skin Lesions.
This file loads the arguments, sets the random seed and tunes a selected task.
"""


# Own Module Imports
from utils import *
from config import *

import argparse
import os
import os.path as osp

import torch
from torch import optim
from torchvision import transforms

import trainer
from datasets import get_dataloader
from models.loss import DetectionCriterion
from models.model import DetectionModel


__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy analysis for smartphone video"
__credits__    = ["Daniel Blackley", "Emannuele Trucco", "Stephen McKenna"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


def main():
    args = arguments()

    num_templates = 25  # aka the number of clusters

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_loader, _ = get_dataloader(args.traindata, args, num_templates,
                                     img_transforms=img_transforms)

    model = DetectionModel(num_objects=1, num_templates=num_templates)
    loss_fn = DetectionCriterion(num_templates)

    # directory where we'll store model weights
    weights_dir = "weights"
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)

    # check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    optimizer = optim.SGD(model.learnable_parameters(args.lr), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=20,
                                          last_epoch=args.start_epoch-1)

    # train and evalute for `epochs`
    for epoch in range(args.start_epoch, args.epochs):
        trainer.train(model, loss_fn, optimizer, train_loader, epoch, device=device)
        scheduler.step()

        if (epoch+1) % args.save_every == 0:
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename="checkpoint_{0}.pth".format(epoch+1), save_path=weights_dir)


if __name__ == "__main__":
    # Loads the arguments from a config file and command line arguments.
    description = "Cost-Sensitive Selective Classification for Skin Lesions Using Bayesian Inference"
    arguments = load_arguments(description)
    log(arguments, "Loaded Arguments:")
    print_arguments(arguments)

    # Sets the default device to be used.
    device = get_device(arguments.cuda)
    log(arguments, f"\nDevice set to {device}")

    # Sets the random seed if specified.
    if arguments.seed != 0:
        set_random_seed(arguments.seed)
        log(arguments, f"Set Random Seed to {arguments.seed}")