# Here, logistic regression models (don't need to use gradient descent etc., we can just use sklearn)

import numpy as np
from tqdm import tqdm

import sklearn.linear_model as linear_model

from data import *

def main():
    n = 200
    d = 100
    noise_std = 0.0

    n_val = 10000

    teacher, x, y = generate_teacher_and_data_logit(n, d, noise_std, 0)
    x_val, y_val = get_data_from_teacher_logit(teacher, n_val, d, noise_std)

    train_loader = create_data_loader(x, y, batch_size=100, shuffle=True)

    n_models = 5
    weight_decay = 1e-4

    errors = []

    hidden_width_list = [100, 250, 500, 1000, 1250, 1500, 2000, 2500, 3000]
    for hidden_width in tqdm(hidden_width_list):
        models = [RandomFeatures(input_dim=d, hidden_width=hidden_width, activation=torch.tanh, seed=i, freeze_first_layer=True) for i in range(n_models)]
        # don't need optimizers, we'll just run skealr
        error_list = []
        
        for model in models:
            hidden_features, labels = model.get_hidden_features(train_loader)
            lr = linear_model.LogisticRegression(C = 1 / weight_decay)
            lr.fit(hidden_features.numpy(), labels.numpy().ravel())
            # fix the 2nd layer weights
            model.set_second_layer(torch.tensor(lr.coef_))
            y_val_pred = model(x_val)
            y_val_pred = (y_val_pred > 0).float()
            error = torch.mean((y_val_pred != y_val).float())
            error_list.append(error.item())
            
        errors.append(error_list)

    errors = np.array(errors)
    
    plt.plot(hidden_width_list, errors.mean(axis = 1))
    plt.title("Error of the logistic regression model as a function of the hidden width")
    plt.xlabel("Hidden width")
    plt.ylabel("Error")
    plt.show()        

if __name__ == "__main__":
    main()