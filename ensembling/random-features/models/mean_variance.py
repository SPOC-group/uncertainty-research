import numpy as np
import torch
import torch.nn as nn

class MeanVarianceModel(nn.Module):
    def __init__(self, input_dim, hidden_width, activation, *, seed = 0, first_layer_weights = None, freeze_first_layer = True, bias_second_layer = True) -> None:
        super().__init__()
        self.activation   = activation
        self.hidden_width = hidden_width
        self.input_dim    = input_dim
        self.output_dim   = 1

        self.first_layer = nn.Linear(input_dim, hidden_width, bias = False)
        # first component of 2nd layer will predict the mean, the 2nd component will predict the variance
        self.second_layer_mean = nn.Linear(hidden_width, self.output_dim, bias = False)
        # NOTE : We should use bias for the second layer variance because otherwise the variance will have constant mean 
        # (due to the Gaussianity of the hidden representation)
        self.second_layer_var = nn.Linear(hidden_width, self.output_dim, bias = bias_second_layer)
        
        with torch.no_grad():
            torch.manual_seed(seed)
            if not first_layer_weights is None:
                self.first_layer.weight.copy_(first_layer_weights)
            else:
                nn.init.normal_(self.first_layer.weight, mean=0, std=1)

            nn.init.normal_(self.second_layer_mean.weight, mean=0, std=1)
            nn.init.normal_(self.second_layer_var.weight, mean=0, std=1)
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
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # need to normalize the hidden activation by input_dim so that the variance of the output is 1
        hidden_representation  = self.activation(self.first_layer(x / np.sqrt(self.input_dim) )) / np.sqrt(self.hidden_width)
        mean = self.second_layer_mean(hidden_representation)
        # pass variance through a softplus to ensure it is positive
        # variance = torch.nn.functional.softplus(self.second_layer_var(hidden_representation))
        variance = (self.second_layer_var(hidden_representation)).pow(2) # square activatn here instead of softplus
        return mean, variance