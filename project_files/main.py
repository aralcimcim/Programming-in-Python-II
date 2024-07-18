import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import warnings
from architecture import MyCNN
from dataset import ImagesDataset
from augmented_dataset import AugmentedImagesDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# challange server: https://apps.ml.jku.at/challenge

def evaluate_model(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            inputs, targets = data[:2]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss += loss_fn(outputs, targets).item()

    loss /= len(loader)
    model.train()
    return loss

def main(
    image_dir,
    labels_file,
    results_path='results',
    network_config: dict = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    n_updates: int = 50000,
    device: str = "cuda"
):

    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    #set seeds
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = ImagesDataset(image_dir)

    #dataset length = 12483
    train_size = 10000
    val_size = 1000
    test_size = 1483
    
    indices = torch.arange(len(dataset))
    training_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    training_set = torch.utils.data.Subset(dataset, training_indices)
    validation_set = torch.utils.data.Subset(dataset, validation_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    #train_loader = DataLoader(AugmentedImagesDataset(training_set), batch_size=16, shuffle=False)
    train_loader_augmented = DataLoader(AugmentedImagesDataset(training_set), batch_size=16, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    print(f"training set size: {len(training_set)}")
    print(f"validation set size: {len(validation_set)}")
    print(f"test set size: {len(test_set)}")

    writer = SummaryWriter(log_dir=os.path.join(results_path, "logs"))

    net = MyCNN(num_classes=20)
    net.to(device)

    loss_fn_type = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    write_stats_at = 1000
    validate_at = 5000
    update = 0
    best_val_loss = np.inf
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    saved_model_file = "model.pth"

    while update < n_updates:
        for data in train_loader_augmented:

            inputs, targets = data[:2]
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = loss_fn_type(outputs, targets)
            loss.backward()
            optimizer.step()

            if (update + 1) % write_stats_at == 0:
                writer.add_scalar("Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(f"Parameters/{name}", param, global_step=update)
                    writer.add_histogram(f"Gradients/{name}", param.grad, global_step=update)

            #evaluate on the validation set
            # TODO
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, val_loader, loss_fn_type, device)
                writer.add_scalar("Loss/validation", scalar_value=val_loss, global_step=update)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(net.state_dict(), saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            update += 1
            if update >= n_updates:
                break
    
    update_progress_bar.close()
    writer.close()
    print("Finished Training!")


if __name__ == "__main__":
    main(
        image_dir='training_data',
        labels_file='training_data/labels.csv',
        results_path='results',
        network_config={
            'n_in_channels': 1,
            'n_hidden_layers': 3,
            'n_kernels': 32,
            'kernel_size': 7
        },
        learning_rate=1e-3,
        weight_decay=1e-4,
        n_updates=50000,
        device='cuda'
    )

