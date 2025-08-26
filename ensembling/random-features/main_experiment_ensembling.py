"""
# IGNORE THIS CODE FOR THE MOMENT
def train_models_same_learned_features(parameters : dict):
    d = parameters["d"]
    n = parameters["n"]
    hidden_width = parameters["hidden_width"]
    teacher = Perceptron(input_dim = d)
    x_train, y_train =  get_data_from_teacher(teacher, n, d, 1.0)
    x_val, y_val    = get_data_from_teacher(teacher, 1000, d, 1.0)
    # train a 1st model 
    model = RandomFeatures(input_dim= d, hidden_width=hidden_width, activation=torch.tanh, seed = 0, freeze_first_layer = False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = parameters["lr"])
    training_loop_single_model(model, x_train, y_train, criterion, optimizer, n_epochs= parameters["num_epochs"])
    
    # copy the 1st layer of the trained model
    # normally the 2nd layer is initialized independently
    models = []
    for i in range(parameters["num_models"]):
        models.append(RandomFeatures(d, hidden_width, torch.tanh, seed = i, freeze_first_layer=True))
        models[-1].set_first_layer(model.first_layer.weight)
    optimizers = [ torch.optim.Adam(model.parameters(), lr = parameters["lr"]) for model in models ]

    result = training_loop_several_models_aggregate_loss(models, x_train, y_train, parameters["criterion"], optimizers, n_epochs=parameters["num_epochs"])
    prediction_variance_history = result["prediction_variance"]
    train_loss_history          = result["loss"]
    train_error_history         = result["error"]

    return {
        "prediction_variance" : prediction_variance_history,
        "train_loss" : train_loss_history,
        "train_error" : train_error_history
    }
"""

import argparse
import pandas as pd

from data import *
from models.random_features import *
from training_functions import *

"""
TODO : Compare the real procedure : 1) train model 2) take Laplace approximation and initialize last layers from it, or 2.b) initialize last layer randomly 
vs
Different random features and train the last layer 
vs 
_Same_ random features from everyone and train the last layer
TODO : Initialiser l'initialisation des 2nd layer avec une Gaussienne et l'approximation de Laplace
"""

# ================== MAIN ==================

def train_models_same_random_features(parameters : dict):
    """
    All the models share the same 1st layer that is taken randomly
    """
    d = parameters["d"]
    n = parameters["n"]
    hidden_width = parameters["hidden_width"]
    teacher = Perceptron(input_dim = d)
    x_train, y_train = get_data_from_teacher(teacher, n, d, 1.0)
    n_val = 200
    x_val, y_val     = get_data_from_teacher(teacher, n_val, d, 1.0)

    activation = parameters["activation"]

    models = [ RandomFeatures(input_dim= d, hidden_width=hidden_width, activation=activation, seed = 0, freeze_first_layer = True) ]
    for i in range(1, parameters["num_models"]):
        # the 2nd layer is initialized differently for each model 
        model_to_add = RandomFeatures(d, hidden_width, activation=activation, seed = i, freeze_first_layer=True, first_layer_weights=models[0].first_layer.weight)
        models.append( model_to_add )
    optimizers = [ torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = parameters["lr"], weight_decay=parameters["pretraining_weight_decay"]) for model in models ]

    for model, optimizer in tqdm(zip(models, optimizers)):
        result = training_loop_single_model(model, x_train, y_train, torch.nn.MSELoss(), torch.nn.MSELoss(), optimizer, n_epochs=5000, show_progress_bar=False)

    result = training_loop_several_models_aggregate_loss(models, x_train, y_train, parameters["criterion"], optimizers, n_epochs=parameters["num_epochs"], x_val=x_val, y_val=y_val)
    prediction_variance_history = result["prediction_variance"]
    train_loss_history          = result["loss"]
    train_error_history         = result["error"]
    val_prediction_variance_history = result["prediction_variance_val"]
    val_error_history            = result["error_val"]

    return {
        "prediction_variance" : prediction_variance_history,
        "prediction_variance_val": val_prediction_variance_history,
        "train_loss" : train_loss_history,
        "train_error" : train_error_history,
        "val_error" : val_error_history
    }

def train_models_different_random_features(parameters : dict):
    d = parameters["d"]
    n = parameters["n"]
    hidden_width = parameters["hidden_width"]
    activation = parameters["activation"]
    teacher = Perceptron(input_dim = d)
    # generate 10 mnodels 
    models = [ RandomFeatures(input_dim = d, hidden_width = hidden_width, activation = activation, seed = i, freeze_first_layer=True) for i in range(parameters["num_models"]) ]
    # generate training data
    x_train, y_train = get_data_from_teacher(teacher, n, d, 1.0)
    n_val = 200
    x_val, y_val     = get_data_from_teacher(teacher, n_val, d, 1.0)

    # train the models
    criterion = parameters["criterion"]
    optimizers = [ torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = parameters["lr"], weight_decay=parameters["pretraining_weight_decay"]) for model in models ]

    for model, optimizer in tqdm(zip(models, optimizers)):
        result = training_loop_single_model(model, x_train, y_train, torch.nn.MSELoss(), torch.nn.MSELoss(), optimizer, n_epochs=5000, show_progress_bar=False)

    result = training_loop_several_models_aggregate_loss(models, x_train, y_train, criterion, optimizers, n_epochs=parameters["num_epochs"], x_val=x_val, y_val=y_val)
    prediction_variance_history = result["prediction_variance"]
    train_loss_history          = result["loss"]
    train_error_history         = result["error"]
    val_error_history            = result["error_val"]
    val_prediction_variance_history = result["prediction_variance_val"]

    return {
        "prediction_variance" : prediction_variance_history,
        "prediction_variance_val" : val_prediction_variance_history,

        "train_loss" : train_loss_history,
        "train_error" : train_error_history,
        "val_error" : val_error_history
    }

def main():
    parser = argparse.ArgumentParser(description='Train models with varying hidden_width.')
    parser.add_argument('--lr', type=float,   help='Learning rate', default=1e-1)
    parser.add_argument('--hidden_width', type=int,   help='Width of the hidden layer', default=100)
    parser.add_argument('--num_models', type=int,   help='Number of models to train', default=20)
    parser.add_argument('--d', type=int,   help='Dimension of the input space', default=50)
    parser.add_argument('--n', type=int,   help='Number of training samples', default=20)
    parser.add_argument('--num_epochs', type=int,   help='Number of epochs', default=5000)
    parser.add_argument('--pretraining_weight_decay', type=float,   help='Weight decay for pretraining', default=1e-4)
    
    args = parser.parse_args()
    
    parameters = {
        'lr': args.lr,
        'hidden_width': args.hidden_width,
        'num_models': args.num_models,
        'd' : args.d,
        'n' : args.n,
        'num_epochs' : args.num_epochs,
        'activation' : torch.tanh,
        'criterion' : AggregateGaussianNLL(),
        'pretraining_weight_decay' : args.pretraining_weight_decay
    }

    print("Parameters are : ")
    for key, value in parameters.items():
        print(f"{key} : {value}")

    result_same_rf = train_models_same_random_features(parameters)
    result_diff_rf = train_models_different_random_features(parameters)

    # plot the variance of the predictions
    plt.clf()
    plt.plot(result_same_rf["prediction_variance"], label="Pred. variance same random features")
    plt.plot(result_diff_rf["prediction_variance"], label="Pred. variance different random features")
    plt.plot(result_same_rf["train_error"], label="Error same random features")
    plt.plot(result_diff_rf["train_error"], label="Error different random features")
    plt.title("Error / prediction variance on training set")
    plt.xlabel("Epoch")
    plt.ylabel("Prediction variance")
    plt.legend()
    plt.savefig(f"error_vs_prediction_variance_train_nb_models={parameters['num_models']}_hidden_width={parameters['hidden_width']}_d={parameters['d']}_n={parameters['n']}.png")

    # plot the prediction variance and MSE error on the validation set
    plt.clf()
    plt.plot(result_same_rf["prediction_variance_val"], label="Pred. variance same random features")
    plt.plot(result_diff_rf["prediction_variance_val"], label="Pred. variance different random features")
    plt.plot(result_same_rf["val_error"], label="Error same random features")
    plt.plot(result_diff_rf["val_error"], label="Error different random features")
    plt.legend()
    plt.savefig(f"error_vs_prediction_variance_val_nb_models={parameters['num_models']}_hidden_width={parameters['hidden_width']}_d={parameters['d']}_n={parameters['n']}.png")
    plt.title("Prediction variance on validation set")

def main_performance_vs_hidden_width():
    """
    From the file optimal_regularization_results.csv, read n,d,hidden_width and optimal_weight_decay. For each value, 
    compute the validation error and validation prediction variance and plot it as a function of hidden_width
    """
    df = pd.read_csv("optimal_regularization_results.csv")
    df = df.groupby(["n", "d", "hidden_width"]).mean().reset_index()

    validation_error_same_rf = []
    validation_error_diff_rf = []
    validation_variance_same_rf = []
    validation_variance_diff_rf = []

    # iterate over the rows of the dataframe
    for i, row in df.iterrows():
        if i >= 5:
            break
        # extract n, d, hidden_width and optimal_weight_decay
        n = row["n"]
        d = row["d"]
        hidden_width = row["hidden_width"]
        weight_decay = row["weight_decay"]

        parameters = {
            'lr': 1e-1,
            'hidden_width': int(hidden_width),
            'num_models': 20,
            'd' : int(d),
            'n' : int(n),
            'num_epochs' : 2000,
            'activation' : torch.tanh,
            'criterion' : AggregateGaussianNLL(),
            'pretraining_weight_decay' : weight_decay
        }

        print(parameters)

        result_diff_rf = train_models_different_random_features(parameters)
        result_same_rf = train_models_same_random_features(parameters)

        validation_error_same_rf.append(result_same_rf["val_error"][-1])
        validation_error_diff_rf.append(result_diff_rf["val_error"][-1])
        validation_variance_same_rf.append(result_same_rf["prediction_variance_val"][-1])
        validation_variance_diff_rf.append(result_diff_rf["prediction_variance_val"][-1])

    print(validation_error_same_rf)
    print(validation_error_diff_rf)
    print(validation_variance_same_rf)
    print(validation_variance_diff_rf)

    # plot the results
    plt.clf()
    plt.plot(df["hidden_width"][:5], validation_error_same_rf, label="Validation error same random features")
    plt.plot(df["hidden_width"][:5], validation_error_diff_rf, label="Validation error different random features")
    plt.plot(df["hidden_width"][:5], validation_variance_same_rf, label="Validation variance same random features")
    plt.plot(df["hidden_width"][:5], validation_variance_diff_rf, label="Validation variance different random features")
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    main()
    # main_performance_vs_hidden_width()