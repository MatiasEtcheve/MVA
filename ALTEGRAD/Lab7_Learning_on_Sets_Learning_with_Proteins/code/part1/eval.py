"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import LSTM, DeepSets
from sklearn.metrics import accuracy_score, mean_absolute_error
from utils import create_test_dataset

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [np.array(X_test[i]).shape[1] for i in range(len(X_test))]
n_samples_per_card = np.array(X_test[0]).shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load("model_deepsets.pth.tar")
deepsets.load_state_dict(checkpoint["state_dict"])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load("model_lstm.pth.tar")
lstm.load_state_dict(checkpoint["state_dict"])
lstm.eval()

# Dict to store the results
results = {"deepsets": {"acc": [], "mae": []}, "lstm": {"acc": [], "mae": []}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):

        ############## Task 6

        ##################
        x_batch = torch.LongTensor(X_test[i])[j : min(n_samples_per_card, j + batch_size)]
        y_pred_deepsets.append(deepsets(x_batch))
        y_pred_lstm.append(lstm(x_batch))
        ##################

    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

    acc_deepsets = accuracy_score(
        y_test[i], np.round(y_pred_deepsets)
    )  # your code here
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)  # your code here
    results["deepsets"]["acc"].append(acc_deepsets)
    results["deepsets"]["mae"].append(mae_deepsets)

    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()

    acc_lstm = accuracy_score(y_test[i], np.round(y_pred_lstm))  # your code here
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)  # your code here
    results["lstm"]["acc"].append(acc_lstm)
    results["lstm"]["mae"].append(mae_lstm)


############## Task 7

##################
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].plot(cards, results["deepsets"]["acc"],label="Deepsets", color="red")
ax[0].plot(cards, results["lstm"]["acc"],label="LSTM", color="blue")
ax[0].legend()
ax[0].set_title("Accuracy on test set")

ax[1].plot(cards, results["deepsets"]["mae"],label="Deepsets", color="red")
ax[1].plot(cards, results["lstm"]["mae"], label="LSTM", color="blue")
ax[1].legend()
ax[1].set_title("MAE on test set")
fig.show()
fig.savefig("test_comparison.png")
##################
