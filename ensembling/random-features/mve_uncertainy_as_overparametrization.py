"""
EXPERIENCES POUR MEAN-VARIANCE ENSEMBLES A LA DEEP ENSEMBLES
TODO : Changer l'activation du 2nd layer qui estimate la variance pour avoir qq chose comparable e.g. au Bayes optimal estimator 
=> pour cela utiliser une activation quadratique
"""

import argparse
import json
import pandas as pd

from data import *
from models.mean_variance import *
from models.random_features import *
from training_functions import *

device = "cpu"

def main_fixed_hidden_width():
    d = 25
    n_list = [50 * i for i in range(1, 11)]
    n_val = 1000
    hidden_width = 50
    noise_std = 0.0

    n_models = 1
    n_epochs = 10000

    mse_list = []

    weight_decay = 0.0

    for n in n_list:
        teacher, x_train, y_train = generate_teacher_and_data(n, d, noise_std = noise_std, seed = 100, device = device)
        train_loader = create_data_loader(x_train, y_train, batch_size=100, shuffle=True)
        x_val, y_val = get_data_from_teacher(teacher, n_val, d, noise_std, device = device)
        
        # bias_second_layer = False if we use the square activation in the second layer
        models     = [MeanVarianceModel(input_dim=d, hidden_width=hidden_width, activation = torch.tanh, seed = i, bias_second_layer=False).to(device) for i in range(n_models) ]
        optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.05, weight_decay = weight_decay) for model in models]
    
        losses     = []
        errors     = []

        # train the models independently
        for model, optimizer in tqdm(zip(models, optimizers)):
            result = training_loop_single_mean_variance_model(model, train_loader, optimizer, n_epochs = n_epochs, data_loader_val = None)
            losses.append(result['loss'])
            errors.append(result['error'])

        # EVALUATE THE UNCERTAINTY VS THE MSE ERROR ON VALIDATION DATA
        means, variances = [], []
        for model in models:
            model.eval()
            mean, variance = model(x_val)
            means.append(mean)
            variances.append(variance)
    
        # compute the variance w.r.t the models of the means array
        means = torch.cat(means, dim = 1)
        variances = torch.cat(variances, dim = 1)

        final_means     = torch.mean(means, dim=1)
        final_variances = torch.mean(variances, dim=1) + torch.var(means, dim = 1)

        mse_list.append( (y_val - final_means).pow(2).mean().item() )
        # final_variance_list.append(final_variances.mean().item())
        # gnll_list.append( -torch.distributions.Normal(final_means, torch.sqrt(final_variances)).log_prob(y_val).mean().item() )

    plt.plot(n_list, mse_list, label = "MSE")
    # plt.plot(n_list, final_variance_list, label = "Predictive variance")
    # plt.plot(n_list, gnll_list, label = "Gaussian NLL")
    plt.legend()
    plt.show()

def main_fixed_n():
    d = 100
    n = 200
    n_val = 1000
    hidden_width_list = np.linspace(50, 500, 10).astype(int)

    noise_std = 1.0
    n_models = 1
    n_epochs = 3000

    teacher, x_train, y_train = generate_teacher_and_data(n, d, noise_std = noise_std, seed = 100, device = device)
    train_loader = create_data_loader(x_train, y_train, batch_size=50, shuffle=True)

    x_val, y_val = get_data_from_teacher(teacher, n_val, d, noise_std, device = device)
    # val_loader = create_data_loader(x_val, y_val, batch_size=50, shuffle=False)

    mse_list = []
    final_variance_list = []

    for hidden_width in hidden_width_list:
        print(hidden_width)
        weight_decay = 0.0
        
        models = [MeanVarianceModel(input_dim=d, hidden_width=hidden_width, activation = torch.tanh, seed = i, bias_second_layer=False).to(device) for i in range(n_models) ]
        optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.05, weight_decay =weight_decay ) for model in models]
    
        losses = []
        errors = []
        prediction_variances = []

        # train the models independently
        for model, optimizer in tqdm(zip(models, optimizers)):
            result = training_loop_single_mean_variance_model(model, train_loader, optimizer, n_epochs = n_epochs, data_loader_val=None)
            losses.append(result['loss'])
            errors.append(result['error'])
            prediction_variances.append(result['prediction_variance'])
            #
            
        # EVALUATE THE UNCERTAINTY VS THE MSE ERROR ON VALIDATION DATA
        means, variances = [], []
        for model in models:
            model.eval()
            mean, variance = model(x_val)
            means.append(mean)
            variances.append(variance)
    
        # compute the variance w.r.t the models of the means array
        means = torch.cat(means, dim = 1)
        variances = torch.cat(variances, dim = 1)

        final_means     = torch.mean(means, dim=1)
        final_variances = torch.mean(variances, dim=1) + torch.var(means, dim = 1)

        mse_list.append( (y_val - final_means).pow(2).mean().item() )
        final_variance_list.append(final_variances.mean().item())

    plt.plot(hidden_width_list, final_variance_list, label = "Predictive variance")
    plt.plot(hidden_width_list, mse_list, label = "MSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_fixed_n()