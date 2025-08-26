import torch
import torch.nn as nn
import numpy as np

class AggregateGaussianNLL(nn.Module):
    """
    Takes as argument the K predictons and returns the NLL of the Gaussian distribution 
    when the mean = mean of the K predictions and the variance = variance of the K predictions
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds : list, target : torch.Tensor) -> torch.Tensor:
        """
        see : https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        preds is a list of size K of the predictions of the model
        """
        stacked_preds = torch.stack(preds, dim=1)
        # compute the average of the K predictions 
        avg_predictions = torch.mean(stacked_preds, dim=1)
        var_predictions = torch.var(stacked_preds, dim=1)
        
        var_predictions.clamp_(min=1e-12)
        loss = nn.GaussianNLLLoss()
        return loss(avg_predictions, target, var_predictions)
    
class AggregateGaussianNLLForMeanVariance(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds : list, target : torch.Tensor) -> torch.Tensor:
        """
        preds is a list of size K of the predictions of the model
        """
        # compute the average of the K predictions 
        stacked_means = torch.stack([p[0] for p in preds], dim=1)
        stacked_vars = torch.stack([p[1] for p in preds], dim=1)

        avg_predictions = torch.mean(stacked_means, dim=1)
        var_predictions = torch.var(stacked_means, dim=1) + torch.mean(stacked_vars, dim=1)

        var_predictions.clamp_(min=1e-12)
        loss = nn.GaussianNLLLoss()
        return loss(avg_predictions, target, var_predictions)

class GaussianNLLForMeanVariance(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds : tuple, target : torch.Tensor) -> torch.Tensor:
        """
        preds is a tuple of size 2 containing the mean and the variance
        """
        mean, variance = preds
        variance.clamp_(min=1e-12)
        loss = nn.GaussianNLLLoss()
        return loss(mean, target, variance) 

class MSEForMeanVariance(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds : tuple, target : torch.Tensor) -> torch.Tensor:
        """
        preds is a tuple of size 2 containing the mean and the variance
        """
        mean, _ = preds
        return (mean - target).pow(2).mean()

class AggregateCRPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, y, mean, variance):
        return variance * ( (y - mean) / variance * torch.erf((y - mean) / (variance * np.sqrt(2))) + 2 / np.sqrt(2 * np.pi) * torch.exp(- (y - mean)**2 / (2 * variance)) - 1 / np.sqrt(np.pi))

    # def forward(self, preds : list, target : torch.Tensor) -> torch.Tensor:
    #     """
    #     see : https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml
    #     """
    #     avg_predictions = torch.mean(torch.stack(preds), dim=0)
    #     var_predictions = torch.var(torch.stack(preds), dim=0)
    #     return self.loss(target, avg_predictions, var_predictions).mean()
    
    def CRPS_new(self, means: torch.Tensor, targets: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        """
        Version of Michele's student
        """
        vars = vars.clone()
        with torch.no_grad():
            vars.clamp_(min=1e-12)
        sigma = torch.sqrt(vars)
        norm_x = ( targets - means)/sigma
        cdf =   0.5 * (1 + torch.erf(norm_x / torch.sqrt(torch.tensor(2))))
        normalization = 1 / (torch.sqrt(torch.tensor(2.0*torch.pi)))
        pdf = normalization * torch.exp(-(norm_x ** 2)/2.0)
        crps = sigma * (norm_x * (2*cdf-1) + 2 * pdf - 1/(torch.sqrt(torch.tensor(torch.pi))))
        return torch.mean(crps)
    
    def forward(self, preds : list, target : torch.Tensor) -> torch.Tensor:
        avg_predictions = torch.mean(torch.stack(preds), dim=0)
        var_predictions = torch.var(torch.stack(preds), dim=0)
        return self.CRPS_new(avg_predictions, target, var_predictions)
    

