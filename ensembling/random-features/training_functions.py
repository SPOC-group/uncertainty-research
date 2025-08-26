from tqdm import tqdm
import torch

from models.random_features import *

# ================== TRAINING FUNCTIONS =========================

def training_loop_single_model(model, data_loader, criterion, error_function, optimizer, n_epochs = 100, show_progress_bar = True):
    loss_history = []
    error_history = []

    model.train()

    for epoch in tqdm(range(n_epochs), disable = not show_progress_bar):
        current_loss = []
        current_error = []
        for x_train, y_train in data_loader:
            optimizer.zero_grad()
            
            predictions = model(x_train)
            
            loss = criterion(predictions, y_train)
            loss.backward(retain_graph = True)
            current_loss.append(loss.item())
            current_error.append( error_function(predictions, y_train).item() )
            optimizer.step()

        loss_history.append( sum(current_loss) / len(current_loss) )
        error_history.append( sum(current_error) / len(current_error) )
        
    return {
        "loss" : loss_history,
        "error" : error_history
    }

def training_loop_single_mean_variance_model(model, data_loader, optimizer, n_epochs = 100, data_loader_val = None):
    loss_history = []
    mse_history = []
    training_variance_history = []

    loss_history_val = []
    mse_history_val = []
    training_variance_history_val = []

    criterion = torch.nn.GaussianNLLLoss(eps=1e-3)

    for epoch in range(n_epochs):
        model.train()

        current_loss = []
        current_mse = []
        current_variance = []
        for x_train, y_train in data_loader:
            optimizer.zero_grad()
            means, variances = model(x_train)
            
            loss = criterion(means, y_train, variances)
            loss.backward(retain_graph = True)
            optimizer.step()
            
            current_loss.append(loss.item())
            current_mse.append((means - y_train).pow(2).mean().item())
            current_variance.append(variances.mean().item())

        loss_history.append(sum(current_loss) / len(current_loss))
        mse_history.append(sum(current_mse) / len(current_mse))
        training_variance_history.append(sum(current_variance) / len(current_variance))

        if not data_loader_val is None:
            # evaluate the predition variance and mse on the validation set
            with torch.no_grad():
                model.eval()
                current_loss_val = []
                current_mse_val = []
                current_variance_val = []
                for x_val, y_val in data_loader_val:
                    means_val, variances_val = model(x_val)
                    loss_val = criterion(means_val, y_val, variances_val)
                    current_loss_val.append(loss_val.item())
                    current_mse_val.append((means_val - y_val).pow(2).mean().item())
                    current_variance_val.append(variances_val.mean().item())
                loss_history_val.append(sum(current_loss_val) / len(current_loss_val))
                mse_history_val.append(sum(current_mse_val) / len(current_mse_val))
                training_variance_history_val.append(sum(current_variance_val) / len(current_variance_val))
        
        
    return {
        "loss" : loss_history,
        "error" : mse_history,
        "prediction_variance" : training_variance_history,
        "loss_val" : loss_history_val,
        "error_val" : mse_history_val,
        "prediction_variance_val" : training_variance_history_val
    }


def training_loop_several_models_aggregate_loss(models : list, data_loader, criterion, optimizer, n_epochs = 100, x_val = None, y_val = None, show_progress_bar = True):
    """
    NOTE : optimizer requires all the parameters of the models
    """
    error_history= [] # error is the Mean square error
    loss_history = []
    pred_variance_history = []

    error_history_val = []
    loss_history_val = []
    pred_variance_history_val = []

    for epoch in tqdm(range(n_epochs), disable = not show_progress_bar):
        current_loss = []
        current_error = []
        current_pred_variance = []

        for x_train, y_train in data_loader:
            predictions = [ ]
            optimizer.zero_grad()
            
            # Forward pass for all models
            for model in models:
                model.train()
                predictions.append(model(x_train))

            predictions_means     = torch.mean(torch.stack(predictions), dim=0)
            predictions_variances = torch.var(torch.stack(predictions), dim=0)

            loss = criterion(predictions, y_train)
            loss.backward(retain_graph = True)

            current_loss.append(loss.item())
            current_error.append((predictions_means - y_train).pow(2).mean().item())
            current_pred_variance.append(predictions_variances.mean().item())

            optimizer.step()
        
        loss_history.append( sum(current_loss) / len(current_loss) )
        error_history.append( sum(current_error) / len(current_error) )
        pred_variance_history.append( sum(current_pred_variance) / len(current_pred_variance) )



        # run on validation set
        if not x_val is None:
            with torch.no_grad():
                for model in models:
                    model.eval()
                predictions_val = [ model(x_val) for model in models ]
                predictions_means_val     = torch.mean(torch.stack(predictions_val), dim=0)
                loss_history_val.append( criterion(predictions_val, y_val) )
                error_history_val.append( (predictions_means_val - y_val).pow(2).mean().item() )
                pred_variance_history_val.append( torch.var(torch.stack(predictions_val), dim=0).mean().item() )

    return {
        "error": error_history,
        "loss" : loss_history,
        "prediction_variance" : pred_variance_history,
        "error_val" : error_history_val,
        "loss_val"  : loss_history_val,
        "prediction_variance_val" : pred_variance_history_val
    }

def evaluate_model(model, x_val, y_val, criterion):
    with torch.no_grad():
        model.eval()
        predictions = model(x_val)
        loss = criterion(predictions, y_val)
    return loss

