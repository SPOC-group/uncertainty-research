# run with python main.py --hidden_depth 5 --total_size 50 --min_width 1 --max_width 4 --n_runs 10 --n_components 25

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm   
from sklearn.model_selection import train_test_split
import argparse
import os 

from laplace import compute_last_layer_hessian, hinge_loss, average_confidence

def save_model_for_width(model, width, args, folder_base="models"):
    """
    Sauvegarde le modèle dans un dossier nommé selon les arguments de la ligne de commande.
    """
    # Crée le nom du dossier à partir des arguments
    folder_name = f"{folder_base}/depth{args.hidden_depth}_size{args.total_size}_runs{args.n_runs}_comp{args.n_components}_minw{args.min_width}_maxw{args.max_width}"
    os.makedirs(folder_name, exist_ok=True)
    # Chemin du fichier
    model_path = os.path.join(folder_name, f"model_width{width}.pt")
    torch.save(model.state_dict(), model_path)

class FiveLayerFCNN(nn.Module):
    def __init__(self, input_dim, hidden_width, hidden_depth=2):
        super(FiveLayerFCNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_width)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_width, hidden_width) for _ in range(hidden_depth)
        ])
        self.output_layer = nn.Linear(hidden_width, 1)
        # Orthogonal initialization
        nn.init.orthogonal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x.squeeze(-1)  # Output shape: (batch,) or scalar

    def get_last_layer(self, x):
        """
        Retourne la sortie de la dernière couche avant l'activation.
        """
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        return x

# Example usage:
# model = FiveLayerFCNN(hidden_width=64)
# output = model(torch.randn(32, 10))  # batch of 32, input dim 10

def get_binary_mnist_pca(n_components=10, batch_size=128, total_size=1000, val_split=0.2):
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.view(-1)  # Flatten
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter for 0 and 1 only
    train_mask = (train_dataset.targets == 0) | (train_dataset.targets == 1)
    test_mask = (test_dataset.targets == 0) | (test_dataset.targets == 1)
    X_train = train_dataset.data[train_mask].numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y_train = train_dataset.targets[train_mask].numpy()
    X_test = test_dataset.data[test_mask].numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y_test = test_dataset.targets[test_mask].numpy()

    # Set labels: +1 for 1, -1 for 0
    y_train = np.where(y_train == 1, 1, -1).astype(np.float32)
    y_test = np.where(y_test == 1, 1, -1).astype(np.float32)

    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    # transform the data to project on the first n_components so that they are distributed as a standard normal
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_train_pca = (X_train_pca - X_train_pca.mean(axis=0)) / X_train_pca.std(axis=0)
    X_test_pca = (X_test_pca - X_test_pca.mean(axis=0)) / X_test_pca.std(axis=0)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # keep only randomly chosen total_size samples for the training set
    indices = np.random.choice(len(X_train_tensor), min(total_size, len(X_train_tensor)), replace=False)
    X_train_tensor = X_train_tensor[indices]
    y_train_tensor = y_train_tensor[indices]

    # Split into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=val_split, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, args, epochs=10, lr=1e-4, device='cpu', patience=2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    min_train_error = 1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = hinge_loss(outputs, y)
            # compute the cross entropy loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            preds = torch.sign(outputs)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        train_error = 1 - (total_correct / total_samples)
        if train_error < min_train_error:
            min_train_error = train_error
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = hinge_loss(outputs, y)
                val_loss += loss.item() * X.size(0)

    # save the model for the current width
    save_model_for_width(model, model.input_layer.out_features, args)
    return min_train_error

def evaluate_model(model, data_loader, device='cpu', n_bins=10):
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.sign(outputs)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_outputs.append(outputs.cpu())
            all_labels.append(y.cpu())
    accuracy = correct / total

    # Calibration (ECE)
    logits = torch.cat(all_outputs).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()
    probs = 1 / (1 + np.exp(-logits))
    labels_bin = (labels == 1).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(labels_bin[mask])
            bin_conf = np.mean(probs[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / total
    return accuracy, ece

def evaluate_laplace(model, data_loader, reg, device='cpu', n_bins=10):
    """
    Compute the new confidence scores with Laplace by computing for each sample the confidence as : 
    reg will be the weight decay from the optimizer 
    """
    # compute 
    model.eval()
    hessian = compute_last_layer_hessian(model, data_loader, reg=reg, device=device)
    hessian_inv = torch.linalg.inv(hessian + 1e-6 * torch.eye(hessian.shape[0]))  # Regularization for numerical stability

    all_outputs = []
    all_labels = []
    all_variances = []

    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            X_ll = model.get_last_layer(X)  # Get the last layer output before activation
            ones = torch.ones(X_ll.shape[0], 1, device=X_ll.device, dtype=X_ll.dtype)  # Bias term for the last layer
            X_ll_bias = torch.cat([X_ll, ones], dim=1)
            outputs = model(X)
            all_outputs.append(outputs.cpu())
            all_labels.append(y.cpu())
            all_variances.append(torch.tensor(np.diag(X_ll_bias @ hessian_inv @ X_ll_bias.T)).cpu())
            total += y.size(0)

    logits = torch.cat(all_outputs).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()
    variances = torch.cat(all_variances).numpy().flatten()

    probs = average_confidence(logits, variances)
    labels_bin = (labels == 1).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(labels_bin[mask])
            bin_conf = np.mean(probs[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / total
    return ece

def generalization_error_vs_width(widths, args, epochs=10, device='cpu', patience=5, n_runs=5, hidden_depth=2, total_size=1000, n_components=10):
    test_errors = []
    test_eces = []
    test_laplace_eces = []
    train_errors = []

    std_test_errors = []
    std_test_eces = []
    std_test_laplace_eces = []
    std_train_errors = []

    for width in tqdm(widths):
        run_test_errors = []
        run_test_eces = []
        run_train_errors = []
        run_test_laplace_eces = []
        for run in range(n_runs):
            train_loader, val_loader, test_loader = get_binary_mnist_pca(total_size=total_size, n_components=n_components)
            model = FiveLayerFCNN(input_dim=n_components, hidden_width=width, hidden_depth=hidden_depth)
            min_train_error = train_model(model, train_loader, val_loader, args, epochs=epochs, lr=1e-4, device=device, patience=patience)
            
            test_acc, test_ece = evaluate_model(model, test_loader, device)
            test_laplace_ece = evaluate_laplace(model, test_loader, reg=0.0, device='cpu', n_bins=10) # so far we hard coded that reg = 0.0
            

            test_error = 1 - test_acc
            
            run_test_errors.append(test_error)
            run_test_eces.append(test_ece)
            run_train_errors.append(min_train_error)
            run_test_laplace_eces.append(test_laplace_ece)

        test_errors.append(np.mean(run_test_errors))
        test_eces.append(np.mean(run_test_eces))
        test_laplace_eces.append(np.mean(run_test_laplace_eces))
        train_errors.append(np.mean(run_train_errors))

        std_test_errors.append(np.std(run_test_errors))
        std_test_eces.append(np.std(run_test_eces))
        std_test_laplace_eces.append(np.std(run_test_laplace_eces))
        std_train_errors.append(np.std(run_train_errors))

    return test_errors, train_errors, test_eces, test_laplace_eces, std_test_errors, std_train_errors, std_test_eces, std_test_laplace_eces 

if __name__ == "__main__":
    # take as command line argument the hidden depth and the total size, and the log of the widths (min and max)
    # use argparse to parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_depth', type=int, default=2)
    parser.add_argument('--total_size', type=int, default=1000)
    parser.add_argument('--min_width', type=float, default=2)
    parser.add_argument('--max_width', type=float, default=4)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--n_components', type=int, default=10)
    args = parser.parse_args()

    # parse the widths as a tuple of two floats
    min_width, max_width = args.min_width, args.max_width
    widths = np.logspace(min_width, max_width, 20, dtype=int)
    # Generalization error vs width experiment
    test_errors, train_errors, test_eces, test_laplace_eces, std_test_errors, std_train_errors, std_test_eces, std_test_laplace_eces = generalization_error_vs_width(widths, args, epochs=10, device='cpu', patience=5, n_runs=args.n_runs, hidden_depth=args.hidden_depth, total_size=args.total_size, n_components=args.n_components)
    
    # TODO : Fill between 
    plt.figure(figsize=(8, 5))
    plt.plot(widths, test_errors, marker='o', label='Generalization error')
    plt.plot(widths, train_errors, marker='s', label='Min training Error')
    plt.fill_between(widths, np.array(test_errors) - np.array(std_test_errors),  np.array(test_errors) + np.array(std_test_errors), alpha=0.2) # transparency 0.2 
    plt.fill_between(widths, np.array(train_errors) - np.array(std_train_errors),  np.array(train_errors) + np.array(std_train_errors), alpha=0.2)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Hidden Layer Width')
    plt.ylabel('Error')
    # plt.title(f'Generalization and Training Error vs Hidden Layer Width (Averaged over {args.n_runs} runs)')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()

    # plot the test ECE
    plt.figure(figsize=(8, 5))
    plt.plot(widths, test_eces, marker='o', label='Test ECE')
    plt.fill_between(widths, np.array(test_eces) - np.array(std_test_eces), np.array(test_eces) + np.array(std_test_eces), alpha = 0.2)
    plt.plot(widths, test_laplace_eces, marker='s', label='Test ECE with Laplace')
    plt.fill_between(widths, np.array(test_laplace_eces) - np.array(std_test_laplace_eces), np.array(test_laplace_eces) + np.array(std_test_laplace_eces), alpha = 0.2)
    plt.xscale('log')
    plt.xlabel('Hidden Layer Width')
    plt.ylabel('ECE')
    # plt.title(f'Test ECE vs Hidden Layer Width (Averaged over {args.n_runs} runs)')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()