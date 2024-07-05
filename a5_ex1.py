import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

            eval_loss.append(np.mean(loss_minibatch))

    return training_loss, eval_loss


# if __name__ == "__main__":
#     from a4_ex1 import SimpleNetwork
#     from dataset import get_dataset
#     torch.random.manual_seed(1234)
#     train_data, eval_data = get_dataset()
#     network = SimpleNetwork(32, [128, 64, 128], 1, True)
#     train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=10,
#     batch_size=16, learning_rate=1e-3, show_progress=True)
#     for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
#         print(f"Epoch: {epoch} --- Train loss: {tl:7.4f} --- Eval loss: {el:7.4f}")