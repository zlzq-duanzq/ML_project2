"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import *
from utils import config
import utils
from sklearn import metrics
from torch.nn.functional import softmax
import torchvision.models

def main():
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="augment",
            batch_size=config("challenge.batch_size"),
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
        )
    
    # Model
    
    model = Challenge()

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=0.01)
    #

    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge...")
    model, start_epoch, stats = restore_checkpoint(model, config("challenge.checkpoint"))

    axes = utils.make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    #TODO: define patience for early stopping
    patience = 5
    curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        #
        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    utils.save_challenge_training_plot()
    utils.hold_training_plot()

    index = np.argmin(np.array(stats)[:,1])
    print(f"epoch: {index}, Training AUROC: {stats[index][5]}, Validation AUROC: {stats[index][2]}")
    #print(f"epoch: {index}, Training AUROC: {stats[index][5]}, Validation AUROC: {stats[index][2]}, Test AUROC: {stats[index][8]}")
    #print(f"Test Accuracy: {stats[index][6]}")

if __name__ == "__main__":
    main()
