"""
EECS 445 - Introduction to Machine Learning
Winter 2021  - Project 2
Predict Challenge
    Runs the challenge model inference on the test dataset and saves the
    predictions to disk
    Usage: python predict_challenge.py --uniqname=<uniqname>
"""

import argparse
import torch
import numpy as np
import pandas as pd
import utils
from dataset import get_challenge
from model.challenge import Challenge
from train_common import *
from utils import config

import utils
from sklearn import metrics
from torch.nn.functional import softmax


def predict_challenge(data_loader, model):
    """
    Runs the model inference on the test set and outputs the predictions
    """
    model_pred = []
    for i, (X, y) in enumerate(data_loader):
        output = model(X)
        predicted = softmax(output.data, dim=1)
        model_pred.append(predicted)
    model_pred = torch.cat(model_pred)
    return model_pred.numpy()


def main(uniqname):
    """Train challenge model."""
    # data loaders
    if check_for_augmented_data("./data"):
        ch_loader, get_semantic_label = get_challenge(
            task="target",
            batch_size=config("challenge.batch_size"), augment = True
        )
    else:
        ch_loader, get_semantic_label = get_challenge(
            task="target",
            batch_size=config("challenge.batch_size"),
        )

    model = Challenge()

    # Attempts to restore the latest checkpoint if exists
    model, _, _ = restore_checkpoint(model, config("challenge.checkpoint"))

    # Evaluate model
    model_pred = predict_challenge(ch_loader, model)

    print("saving challenge predictions...\n")
    #model_pred = [get_semantic_label(p) for p in np.argmax(model_pred,axis=1)]
    pd_writer = pd.DataFrame(model_pred, columns=["golden_retriever","collie"])
    pd_writer.to_csv(uniqname + ".csv", index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniqname", required=True)
    args = parser.parse_args()
    main(args.uniqname)
