"""
EXPERIENCES POUR MEAN-VARIANCE ENSEMBLES A LA DEEP ENSEMBLES
"""

import argparse
import json
import pandas as pd
import sklearn.linear_model as linear_model

from data import *
from models.mean_variance import *
from models.random_features import *
from training_functions import *

device = "cpu"

def main_fixed_hidden_width():
    """
    Here, only fix the random features and train the 2nd layer explicitely
    """
    d            = 10
    n_list       = np.linspace(600, 2000, 100).astype(int)
    n_val        = 10000
    hidden_width = 500
    noise_std    = 0.0

    n_models = 1

    mse_list = []
    weight_decay = 0.0

    for n in tqdm(n_list):
        teacher, x_train, y_train = generate_teacher_and_data(n, d, noise_std = noise_std, seed = 100, device = device)
        train_loader = create_data_loader(x_train, y_train, batch_size=100, shuffle=True)
        x_val, y_val = get_data_from_teacher(teacher, n_val, d, noise_std, device = device)
        
        models     = [RandomFeatures(input_dim=d, hidden_width=hidden_width, activation = torch.tanh, seed = i).to(device) for i in range(n_models) ]
    
        for model in models:
            features, labels = model.get_hidden_features(train_loader)
            lr = linear_model.Ridge(alpha = weight_decay, fit_intercept=False)
            lr.fit(features.numpy(), labels.numpy())
            model.set_second_layer(torch.tensor(lr.coef_))

        # EVALUATE THE UNCERTAINTY VS THE MSE ERROR ON VALIDATION DATA
        means, variances = [], []
        for model in models:
            model.eval()
            mean = model(x_val)
            means.append(mean)
    
        # compute the variance w.r.t the models of the means array
        means = torch.cat(means, dim = 1)

        final_means     = torch.mean(means, dim=1)
        mse_list.append( (y_val - final_means).pow(2).mean().item() )

    plt.plot(n_list, mse_list, label = "MSE")
    plt.legend()
    plt.show()

def main_fixed_n():
    """
    NOTE : Ici on observe bien un double-descent dans le mean square error; de plus en utulisant l'identite comme activation, la predictive 
    variance se reduit a 0 dans le overparametrized regime 
    """
    d = 100
    n = 200
    n_val = 1000
    hidden_width_list = np.linspace(50, 1000, 50).astype(int)
    noise_std = 0.0

    n_models = 10

    teacher, x_train, y_train = generate_teacher_and_data(n, d, noise_std = noise_std, seed = 100, device = device)
    train_loader              = create_data_loader(x_train, y_train, batch_size=50, shuffle=True)
    x_val, y_val              = get_data_from_teacher(teacher, n_val, d, noise_std, device = device)

    mse_list = []
    predictive_variance_list = []

    weight_decay = 0.0

    for hidden_width in tqdm(hidden_width_list):
        models = [RandomFeatures(input_dim=d, hidden_width=hidden_width, activation = nn.Identity(), seed = i).to(device) for i in range(n_models) ]
        
        for model in models:
            # train the models manually 
            features, labels = model.get_hidden_features(train_loader)
            lr = linear_model.Ridge(alpha = weight_decay, fit_intercept=False)
            lr.fit(features.numpy(), labels.numpy())
            model.set_second_layer(torch.tensor(lr.coef_))
    
        # EVALUATE THE UNCERTAINTY VS THE MSE ERROR ON VALIDATION DATA
        means = []
        for model in models:
            model.eval()
            mean = model(x_val)
            means.append(mean)
    
        # compute the variance w.r.t the models of the means array
        means = torch.cat(means, dim = 1)
        final_means     = torch.mean(means, dim=1)

        mse_list.append( (y_val - final_means).pow(2).mean().item() )
        predictive_variance_list.append( torch.mean(torch.var(means, dim = 1)).item() )

    plt.plot(hidden_width_list, mse_list, label = "MSE")
    plt.legend()
    plt.show()

    plt.plot(hidden_width_list, predictive_variance_list, label = "Predictive variance")
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    # main_fixed_hidden_width()
    main_fixed_n()