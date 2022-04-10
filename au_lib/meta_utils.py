import os
import json
import numpy as np
import pickle
import argparse
import random
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def compute_label_frequency(label_path):
    # load label
    with open(label_path, 'rb') as f:
        _, labels = pickle.load(f)
    f.close()
    labels = np.concatenate(labels, axis=0)
    all_labelcnt = np.sum(labels, axis=0)

    return all_labelcnt / labels.shape[0]


def compute_class_frequency(label_path):
    # load label
    with open(label_path, 'rb') as f:
        _, labels = pickle.load(f)
    f.close()
    labels = np.concatenate(labels, axis=0)
    all_labelcnt = np.sum(labels, axis=0)

    return all_labelcnt / all_labelcnt.sum()