from rob import SVMParamterEstimator, ModelArgs

import torch
import matplotlib.pyplot as plt


def load_model():
    model_config = {
        "num_particles": 64,
        "input_size": 20,
        "hidden_dimension": 15 
    }
    model = SVMParamterEstimator(model_config)
    model.load_state_dict(torch.load("./models/pfrnn.pt"))
    return model


def load_data():
    with open("xs_test.pt", 'rb') as f:
        xs_test = torch.load(f)
    with open("ys_test.pt", 'rb') as f:
        ys_test = torch.load(f)
    return xs_test, ys_test


if __name__=="__main__":
    # First we need to load our previously trained model
    model = load_model()

    # and init our model args (to default values in this case)
    model_args = ModelArgs()

    # we then load our previously generated eval data
    xs_test, ys_test = load_data()

    # we run our model on the eval data to get its predictions
    ys_pred, particle_pred = model.forward(xs_test)

    # convert to numpy and reshape
    ys_pred_np = ys_pred.detach().numpy()
    ys_test_np = ys_test.detach().numpy()

    ys_pred_np = ys_pred_np.squeeze().flatten()
    ys_test_np = ys_test_np.squeeze().flatten()

    # finally we plot this on a histogram to try and visualize
    # systematic error in the predictor
    plt.hist2d(ys_pred_np, ys_test_np, bins=(25,25), cmap=plt.cm.jet)
    plt.show()
    