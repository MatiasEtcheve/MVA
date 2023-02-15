"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1

    ##################
    X_train = []
    y_train = []
    for _ in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        nodes = np.random.choice(10, card, replace=True)
        nodes = [0 for _ in range(10 - len(nodes))] + list(nodes)
        X_train.append(nodes)
        y_train.append(np.sum(nodes))
    ##################

    return X_train, y_train


def create_test_dataset():

    ############## Task 2

    ##################
    X_test = []
    y_test = []
    for size in range(5, 101, 5):
        X_sample = []
        y_sample = []
        for _ in range(10000):
            x = np.random.randint(1, 11, size=size)
            X_sample.append(x)
            y_sample.append(np.sum(x))
        X_test.append(np.array(X_sample))
        y_test.append(np.array(y_sample))
    ##################

    return X_test, y_test
