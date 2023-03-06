from rob import SVMParamterEstimator, ModelArgs

import torch
import matplotlib.pyplot as plt


def load_model(model_path):
    model_config = {
        "num_particles": 64,
        "input_size": 20,
        "hidden_dimension": 50 
    }
    model = SVMParamterEstimator(model_config)
    model.load_state_dict(torch.load(model_path))
    return model


def load_data():
    with open("xs_test.pt", 'rb') as f:
        xs_test = torch.load(f)
    with open("ys_test.pt", 'rb') as f:
        ys_test = torch.load(f)
    return xs_test, ys_test


if __name__=="__main__":
    # # First we need to load our previously trained model
    # model1 = load_model("./models/pfrnn_epoch_0.pt")
    # model2 = load_model("./models/pfrnn.pt")

    # model1params = list(model1.parameters())
    # model2params = list(model2.parameters())

    # for i in [0, 1, 5]:
    #     #print(torch.mean(model1params[i].data - model2params[i].data), i)
    #     print(model1params[i].data.shape)
    # 1/0

    # First we need to load our previously trained model
    model = load_model("./models/pfrnn.pt")

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
    plt.hist2d(ys_pred_np, ys_test_np, bins=(25,25), cmap=plt.cm.pink)
    plt.show()
    