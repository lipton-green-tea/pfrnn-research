from stochastic_volatility import SVL1, SVL1Paramters
from rob import SVMParamterEstimator, ModelArgs

import math
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    # step 1: generate data
    # 
    # we will create several series for our dataset,
    # each using the same paramaters.
    # We will then split each of the series into a sliding window
    #
    # (note) in later experiments we may also need to add static
    # constants in

    def normalize(xs):
        return (xs - np.min(xs)) / (np.max(xs) - np.min(xs))

    # training config
    config = {
        "samples": 1000,
        "sequence_length": 200,
        "window_size": 10,  # implement the components needed for this in the model
        "train_test_split": 0.8,
        "epochs": 5,
        "batch_size": 200
    }

    sv_parameters = SVL1Paramters(
        alpha=-0.00192640,
        phi=0.972,
        rho=-0.3179,
        sigma=0.1495,
        initial_innovation=0.1,
        initial_volatility=0.1
    )

    # initialize model args to default values 
    model_args = ModelArgs()

    xs = []
    ys = []

    # generate volatility data using the SVL1 model
    for s in range(config["samples"]):
        volatility, innovations = SVL1.generate_data(config["sequence_length"], sv_parameters)

        # normalize values to between 0 and 1
        volatility = normalize(volatility)
        innovations = normalize(innovations)

        # below we create several windows of observations
        window_size = 10
        windows = ( # add arrays together to create sub_windows
            np.expand_dims(np.arange(window_size), 0) +  # time offsets
            np.expand_dims(np.arange(len(innovations) - window_size), 0).T  # start times
        )
        innovation_windows = innovations[windows]

        # get rid of the volatilies for which we do not have enough previous values for
        volatility = volatility[window_size:]

        # reshape volatility to have one output per window 
        volatility = volatility.reshape(len(volatility), 1, )

        # finally we add the arrays to our dataset
        xs.append(innovation_windows)
        ys.append(volatility)

    # data reshaping/formatting
    # 1. convert xs and ys to numpy arrays
    # 2. ensure the types are float32
    # 3. split into test and trai4
    # 3. convert them to pytorch tensors
    xs = np.array(xs)
    ys = np.array(ys)

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    split = round(len(xs) * config["train_test_split"])
    indices = np.random.permutation(len(xs))
    train_indices, test_indices = indices[0:split], indices[split:]
    xs_train, xs_test = xs[train_indices], xs[test_indices]
    ys_train, ys_test = ys[train_indices], ys[test_indices]

    xs_train = torch.from_numpy(xs_train)
    ys_train = torch.from_numpy(ys_train)
    xs_test = torch.from_numpy(xs_test)
    ys_test = torch.from_numpy(ys_test)

    xs_train.type(torch.FloatTensor)
    ys_train.type(torch.FloatTensor)
    xs_test.type(torch.FloatTensor)
    ys_test.type(torch.FloatTensor)


    # step 2: create the model and optimizer
    # 
    # we create a PF-RNN
    # in particular we use the model specified in the 'rob.py' file
    # 
    # we also create an optimizer

    model = SVMParamterEstimator()
    optimizer = torch.optim.RMSprop(
            model.parameters(), lr=0.005)

    # step 3: train the model
    # 
    # we perform a number of epochs.
    # at each epoch we:
    # 1. set the model to train
    # 2. train the model over successive iterations
    # 3. eval the model

    batch_size = config["batch_size"]

    loss_per_epoch = []

    for e in range(config["epochs"]):
        model.train()

        iterations = math.ceil(len(xs_train) / config["batch_size"])
        for i in range(iterations):

            xs_batch = xs_train[i * batch_size:(i + 1) * batch_size]
            ys_batch = ys_train[i * batch_size:(i + 1) * batch_size]

            # reset all gradients that have been built up on the model
            # between iterations
            model.zero_grad()

            # perform 1 step of gradient descent 
            loss, log_loss, particle_pred = model.step(xs_batch,ys_batch,model_args)
            print(loss)
            loss.backward()
            optimizer.step()

        # we now evaluate the model using our eval 
        model.zero_grad()
        loss, log_loss, particle_pred = model.step(xs_test, ys_test, model_args)
        loss_per_epoch.append(loss.detach().numpy())

    
    # step 4: draw the graphs
    # 
    # we want to display our results now
    # we draw a number of graphs to show these including
    # 1. a graph of the loss over epochs
    # 2. a graph showing the model fit to data

    print(loss_per_epoch)

    # we will now fit the model to some data and plot the result
    print(xs_test[-1:].shape)
    ys_pred, particle_pred = model.forward(xs_test[-1:])

    # convert to numpy arrays
    ys_pred = ys_pred.detach().numpy()
    ys_true = ys_test.detach().numpy()

    print(ys_pred.shape)
    print(ys_true.shape)

    # flatten into a 1D array
    ys_pred = ys_pred.reshape((len(ys_pred), ))
    ys_true = ys_test[-1].reshape((len(ys_test[-1], )))


    plt.plot(ys_pred, color="orange")
    plt.plot(ys_true, color="blue")
    plt.show()
    





