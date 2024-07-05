import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def training_loop(network: nn.Module,
                  train_data: Dataset,
                  eval_data: Dataset,
                  num_epochs: int, 
                  batch_size: int,
                  learning_rate: float,
                  show_progress: bool = False) -> tuple[list, list]:
    
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size)

    training_loss = []
    eval_loss = []
    eval_loss_min = float("inf")
    eval_not_improved = 0

    for epoch in range(num_epochs):
        network.train()
        loss_minibatch = []
        for inputs, targets in tqdm(train_loader) if show_progress else train_loader:
            outputs = network(inputs)
            loss = nn.functional.mse_loss(outputs.flatten(), targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_minibatch.append(loss.item())

        training_loss.append(np.mean(loss_minibatch))

        network.eval()
        with torch.no_grad():
            loss_minibatch = []
            for inputs, targets in eval_loader:
                outputs = network(inputs)
                loss = nn.functional.mse_loss(outputs.flatten(), targets)
                loss_minibatch.append(loss.item())

            eval_loss_mean = np.mean(loss_minibatch)
            eval_loss.append(eval_loss_mean)

            if eval_loss_mean < eval_loss_min:
                eval_loss_min = eval_loss_mean
                eval_not_improved = 0
            else:
                eval_not_improved += 1

            # early stopping condition
            if eval_not_improved == 7:
                print(f"Early stopping @ Epoch {epoch} | Loss: {eval_loss_min}")
                break

    return training_loss, eval_loss

def plot_losses(train_losses: list, eval_losses: list):
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(eval_losses, label="Eval loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    #plt.show()
    plt.savefig("epoch_loss.pdf")


# if __name__ == "__main__":
#     from a4_ex1 import SimpleNetwork
#     from dataset import get_dataset
#     torch.random.manual_seed(1234)
#     train_data, eval_data = get_dataset()
#     network = SimpleNetwork(32, [128, 128, 128], 1, True)
#     train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100, batch_size=16, learning_rate=1e-3)
#     plot_losses(train_losses, eval_losses)