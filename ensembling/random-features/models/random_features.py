"""
Fichier contenant la classe RandomFeatures
NOTES : The behaviour of our model (like how the prediction variance converges to 0, explodes or not) seems
dependent on the learning rate 
"""

# remove this import when not needed anymore
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class Perceptron(nn.Module):
    def __init__(self, input_dim, *, seed = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, 1, bias = False)

        torch.manual_seed(seed)
        nn.init.normal_(self.layer1.weight, mean=0, std=1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layer1(x / np.sqrt(self.input_dim)) 
    
    def forward_with_noise(self, x : torch.Tensor, noise_std : torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return y + noise_std * torch.randn_like(y)

class RandomFeatures(nn.Module):
    def __init__(self, input_dim, hidden_width, activation, *, seed = 0, first_layer_weights = None, freeze_first_layer = True) -> None:
        super().__init__()
        self.activation = activation
        self.hidden_width = hidden_width
        self.input_dim = input_dim

        self.first_layer = nn.Linear(input_dim, hidden_width, bias = False)
        self.second_layer = nn.Linear(hidden_width, 1, bias = False)


        with torch.no_grad():
            torch.manual_seed(seed)
            if not first_layer_weights is None:
                self.first_layer.weight.copy_(first_layer_weights)
            else:
                nn.init.normal_(self.first_layer.weight, mean=0, std=1)
            nn.init.normal_(self.second_layer.weight, mean=0, std=1)

            # Freeze the 1st layer
            self.freeze_first_layer = freeze_first_layer
            if freeze_first_layer:
                for param in self.first_layer.parameters():
                    param.requires_grad = False

    def set_first_layer(self, weights : torch.Tensor) -> None:
        """
        Will be used to define an ensemble of models with the same 1st layer 
        """
        with torch.no_grad():
            self.first_layer.weight.copy_(weights)
        
        if self.freeze_first_layer:
            for param in self.first_layer.parameters():
                param.requires_grad = False

    def set_second_layer(self, weights : torch.Tensor) -> None:
        with torch.no_grad():
            self.second_layer.weight.copy_(weights)

    def forward_first_layer(self, x : torch.Tensor) -> torch.Tensor:
        return self.activation(self.first_layer(x / np.sqrt(self.input_dim)))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # need to normalize the hidden activation by input_dim so that the variance of the output is 1
        return self.second_layer( self.forward_first_layer(x) / np.sqrt(self.hidden_width)) 

    def get_hidden_features(self, data_loader):
        hidden_features_list = []
        labels_list = []
        self.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                hidden_features = self.forward_first_layer(inputs)
                hidden_features_list.append(hidden_features)
                labels_list.append(labels)
        
        hidden_features = torch.cat(hidden_features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return hidden_features, labels
