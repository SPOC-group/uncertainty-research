import argparse
import json

import optuna

from models.random_features import *
from models.mean_variance import *
from custom_losses import *
from training_functions import *
from data import *

# ================== HYPERPARAMETER OPTIMIZATION ==================

def optimize_weight_decay(model_factory, x_train, y_train, x_val, y_val, criterion, error_function, *, n_trials = 50, lr = 1e-2, low_weight_decay = 1e-4, high_weight_decay = 1.0):
    """
    model_factory : callable that returns a model without any argument
    by default, criterion is MSELoss
    """
    data_loader = create_data_loader(x_train, y_train, batch_size=32, shuffle=True)
    
    def objective(trial):
        weight_decay = trial.suggest_loguniform("weight_decay", low_weight_decay, high_weight_decay)
        model = model_factory()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, weight_decay = weight_decay)
        
        training_loop_single_model(model, data_loader, criterion, error_function, optimizer, n_epochs = 2000, show_progress_bar = False)
        loss = evaluate_model(model, x_val, y_val, error_function)
        return loss.item()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def main():
    """
    take n, hidden_width, d as argument and compute the optimal regularization parameter
    """
    parser = argparse.ArgumentParser(description='Optimize the learning rate')
    parser.add_argument('--n', type=int, help='Number of samples', default=100)
    parser.add_argument('--d', type=int, help='Dimension of the input', default=50)
    parser.add_argument('--hidden_width', type=int, help='Width of the hidden layer', default=100)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-2)
    parser.add_argument('--n_trials', type=int, help='Number of trials', default=50)
    parser.add_argument('--save_file', type=str, help='File to save the results', default=None)
    parser.add_argument('--model_class', type=str, help='Model class', default="RandomFeatures")
    parser.add_argument('--freeze_first_layer', type=bool, help='Freeze the first layer', default=True)


    args = parser.parse_args()

    n = args.n
    d = args.d
    hidden_width = args.hidden_width
    lr = args.lr
    n_trials = args.n_trials

    teacher = Perceptron(input_dim = d)
    x_train, y_train = get_data_from_teacher(teacher, n, d, 1.0)
    x_val, y_val = get_data_from_teacher(teacher, 1000, d, 1.0)

    model_class = globals()[args.model_class]
    def model_factory(seed = 0):
        return model_class(input_dim = d, hidden_width = hidden_width, activation = torch.tanh, seed = seed, freeze_first_layer = args.freeze_first_layer)
    
    # for both type of models, the error is the mean square error 
    if model_class == RandomFeatures:
        criterion = torch.nn.MSELoss()
        error_function = lambda predictions, y: (predictions - y).pow(2).mean()
    elif model_class == MeanVarianceModel:
        criterion = GaussianNLLForMeanVariance()
        error_function = lambda predictions, y: (predictions[0] - y).pow(2).mean()

    optimal_w_d = optimize_weight_decay(model_factory, x_train, y_train, x_val, y_val, criterion, error_function, n_trials = n_trials, lr = lr, low_weight_decay=1e-4, high_weight_decay=0.1)['weight_decay']
    args = vars(args)
    args["optimal_weight_decay"] = optimal_w_d
    print(optimal_w_d)
    
    # save n, d and hidden_width and optimal_weight_decay
    #append it to the file
    if args["save_file"] is not None:
        with open(args["save_file"], 'w') as f:
            json.dump(args, f, indent=4)

if __name__ == "__main__":
    result = main()