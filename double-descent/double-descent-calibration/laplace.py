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

def hinge_loss(outputs, targets):
    # outputs: (batch,), targets: (batch,)
    return torch.mean(torch.clamp(1 - outputs * targets, min=0))

def compute_last_layer_hessian(model, train_loader, reg=0.0, device='cpu'):
    """
    Calcule la Hessienne (Laplace) de la loss d'entraînement (avec régularisation L2)
    uniquement par rapport aux paramètres de la dernière couche du modèle.
    Args:
        model: modèle PyTorch entraîné
        train_loader: DataLoader du set d'entraînement
        reg: coefficient de régularisation L2 (float)
        device: 'cpu' ou 'cuda'
    Returns:
        hessian: matrice Hessienne (numpy array)
    """
    model.eval()
    model.to(device)
    params = [p for p in model.output_layer.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    hessian = torch.zeros((n_params, n_params), device=device, dtype=torch.float32)
    # On accumule la loss totale sur tout le dataset
    total_loss = 0.0
    total_samples = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = hinge_loss(outputs, y)
        total_loss += loss * X.size(0)
        total_samples += X.size(0)
    avg_loss = total_loss / total_samples
    # Ajout régularisation L2
    if reg > 0.0:
        l2 = 0.0
        for p in params:
            l2 = l2 + (p**2).sum()
        avg_loss = avg_loss + 0.5 * reg * l2
    # Calcul du gradient
    grad = torch.autograd.grad(avg_loss, params, create_graph=True, allow_unused=True)
    grad_flat = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(grad, params)])
    # Calcul de la Hessienne colonne par colonne
    for i in range(n_params):
        grad2 = torch.autograd.grad(grad_flat[i], params, retain_graph=True, allow_unused=True)
        grad2_flat = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(grad2, params)])
        hessian[i] = grad2_flat
    return hessian.cpu()

def average_confidence(logit, variance):
    """
    fn noisy_sigmoid_likelihood(z : f64, noise_std : f64) -> f64 {
    // The exact version of the likelihood is a bit too slow, let's use an approximate form 
    // let integrand = |xi : f64| -> f64 { logistic::logistic( xi * noise_std + z ) * (- xi*xi / 2.0).exp() / (2.0 * PI).sqrt() };
    // return integral::integrate(integrand, (-LOGIT_QUAD_BOUND, LOGIT_QUAD_BOUND), integral::Integral::G30K61(GK_PARAMETER));
    // when noise_std = 0.0, normally it's the "correct" rigorous expression
    return logistic::logistic( z / (1.0 + (LOGIT_PROBIT_SCALING * noise_std).powi(2) ).sqrt() );
}

    """
    LOGIT_PROBIT_SCALING = 0.5875651988237005**2
    sigmoid = lambda x : 1 / (1 + np.exp(-x))
    return sigmoid(logit / np.sqrt(1.0 + LOGIT_PROBIT_SCALING * variance))