from models.random_features import *
import torch
from torch.utils.data import Dataset, DataLoader

def get_data_from_teacher(teacher : Perceptron, n : int, d : int, noise_std : float, device = "cpu") -> tuple:
    x = torch.randn(n, d)
    y = teacher.forward_with_noise(x, noise_std)
    return x, y

def get_data_from_teacher_logit(teacher : Perceptron, n : int, d : int, device = "cpu") -> tuple:
    x = torch.randn(n, d)
    y = teacher.forward(x)
    # generate y from the logit model
    probas = torch.sigmoid(y)
    y = torch.bernoulli(probas)
    return x, y

def generate_teacher_and_data(n : int, d : int, noise_std : float, seed = 0, device = "cpu") -> tuple:
    teacher = Perceptron(input_dim = d, seed = seed)
    x, y = get_data_from_teacher(teacher, n, d, noise_std, device = device)
    return teacher, x, y

def generate_teacher_and_data_logit(n : int, d : int, seed = 0, device = "cpu") -> tuple:
    teacher = Perceptron(input_dim = d, seed = seed)
    x, y = get_data_from_teacher_logit(teacher, n, d, device = device)
    return teacher, x, y

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (numpy.ndarray or torch.Tensor): Input features.
            y (numpy.ndarray or torch.Tensor): Target values.
        """
        if isinstance(X, torch.Tensor):
            self.X = X
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
        
        if isinstance(y, torch.Tensor):
            self.y = y
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loader(X, y, batch_size=32, shuffle=True):
    dataset = SyntheticDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader