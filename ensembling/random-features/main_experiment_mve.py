"""
EXPERIENCES POUR MEAN-VARIANCE ENSEMBLES A LA DEEP ENSEMBLES
"""

import argparse
import pandas as pd

from data import *
from models.mean_variance import *
from models.random_features import *
from training_functions import *

device = "cpu"

def main():
    d = 100
    n = 1000
    n_val = 1000
    hidden_width = 500
    noise_std = 1.0
    weight_decay = 0.0013002913416355233

    n_models = 10
    teacher, x_train, y_train = generate_teacher_and_data(n, d, noise_std = noise_std, seed = 100, device = device)
    train_loader = create_data_loader(x_train, y_train, batch_size=50, shuffle=True)

    x_val, y_val = get_data_from_teacher(teacher, n_val, d, noise_std, device = device)
    val_loader = create_data_loader(x_val, y_val, batch_size=50, shuffle=False)

    models = [MeanVarianceModel(input_dim=d, hidden_width=hidden_width, activation=torch.tanh, seed = i).to(device) for i in range(n_models)  ]
    optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.05, weight_decay =weight_decay ) for model in models]
    
    losses = []
    errors = []
    prediction_variances = []

    prediction_variances_val = []
    errors_val = []
    losses_val = []

    # train the models independently
    for model, optimizer in tqdm(zip(models, optimizers)):
        result = training_loop_single_mean_variance_model(model, train_loader, optimizer, n_epochs = 5000, data_loader_val=None)
        losses.append(result['loss'])
        errors.append(result['error'])
        prediction_variances.append(result['prediction_variance'])
        #
        losses_val.append(result['loss_val'])
        errors_val.append(result['error_val'])
        prediction_variances_val.append(result['prediction_variance_val'])

    # plot the losses
    for loss in losses:
        plt.plot(loss)
    plt.title("Training loss for Gaussian NLL loss")
    plt.show()

    # plot the errors
    for error in errors:
        plt.plot(error)
    plt.title("Training error for MSE")
    plt.show()

    # plot the prediction variances
    for variance in prediction_variances:
        plt.plot(variance)
    plt.title("Training prediction variance")
    plt.show()

    # plot the validation losses
    for loss in losses_val:
        plt.plot(loss)
    plt.title("Validation loss for Gaussian NLL loss")
    plt.show()

    # plot the validation errors
    for error in errors_val:
        plt.plot(error)
    plt.title("Validation error for MSE")
    plt.show()

    # plot the validation prediction variances
    for variance in prediction_variances_val:
        plt.plot(variance)
    plt.title("Validation prediction variance")
    plt.show()

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

    print(final_means.shape, final_variances.shape, y_val.shape)

    mse = (final_means - y_val).pow(2).mean().item()
    uncertainty = final_variances.mean().item()
    print(f"MSE on validation data : {mse}")
    print(f"Uncertainty on validation data : {uncertainty}")

    plt.scatter(final_variances.detach().numpy(), (final_means - y_val[:, 0]).pow(2).detach().numpy())
    plt.xlabel("Uncertainty")
    plt.ylabel("MSE")
    plt.title("Uncertainty vs MSE on validation data")
    plt.show()


if __name__ == "__main__":
    main()