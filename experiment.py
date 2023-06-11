from stochastic_volatility import SVL1, SVL1Paramters, HarveySV, HarveySVParamters
from rob import SVMParamterEstimator, ModelArgs
from harvey_sv_model import HarveySVPF
from small_pfrnn_model import SmallPFRNNModel
from lstm_model import LSTM1

import os
import math
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


if __name__=="__main__":

    # start by clearing the cache to prevent GPU from running out
    # of memory
    torch.cuda.empty_cache()

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
        "samples": 500,
        "sequence_length": 300,
        "window_size": 1,
        "train_test_split": 0.9,
        "epochs": 3, # set to 0 if you don't want to train the model
        "batch_size": 25,
        "learning_rate": 0.0005,
        "load_model_from_previous": False,
        "load_data_from_previous": False,
        "save_models": True,
        "base_path": "./models/small_pfrnn",
        "model_path": "./models/small_pfrnn_0.pt",
        "use_gpu": False
    }

    # sv_parameters = SVL1Paramters(
    #     alpha=-0.00192640,
    #     phi=0.972,
    #     rho=-0.3179,
    #     sigma=0.1495,
    #     initial_innovation=0.5,
    #     initial_volatility=0.5
    # )

    sv_parameters = HarveySVParamters(
        mu=2. * np.log(.7204), 
        phi=.9807, 
        tau=0.1489
    )

    # initialize model args to default values 
    model_args = ModelArgs(
        l1_weight=0.5,
        l2_weight=0.5,
    )
    model_config = {
        "num_particles": 128,
        "input_size": config["window_size"],
        "hidden_dimension": 1
    }

    # here we either load or generate our dataset
    if config["load_data_from_previous"] and \
       os.path.isfile("xs_train.pt"):  # if the xs_train tensor file exists assume the others do too
        # we load our tensors from their files
        with open("xs_train.pt", 'rb') as f:
            xs_train = torch.load(f)
        with open("ys_train.pt", 'rb') as f:
            ys_train = torch.load(f)
        with open("xs_test.pt", 'rb') as f:
            xs_test = torch.load(f)
        with open("ys_test.pt", 'rb') as f:
            ys_test = torch.load(f)
    else:  # generate volatility data using the SVL1 model
        start_time = time.time()

        xs = []
        ys = []

        for s in range(config["samples"]):
            volatility, innovations = HarveySV.generate_data(config["sequence_length"], sv_parameters)

            # normalize values to between 0 and 1
            #volatility = normalize(volatility)
            #innovations = normalize(innovations)
            
            # convert to numpy arrays
            volatility = np.array(volatility)
            innovations = np.array(innovations)

            # below we create several windows of observations
            window_size = config["window_size"]
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

        print(f"time taken to generate data: {time.time() - start_time}")

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

        # save all the tensors so they can be loaded instead
        # of generated next time
        with open("xs_train.pt", 'wb') as f:
            torch.save(xs_train, f)
        with open("ys_train.pt", 'wb') as f:
            torch.save(ys_train, f)
        with open("xs_test.pt", 'wb') as f:
            torch.save(xs_test, f)
        with open("ys_test.pt", 'wb') as f:
            torch.save(ys_test, f)

    print(xs_train.shape)


    # step 2: create the model and optimizer
    # 
    # we create a PF-RNN
    # in particular we use the model specified in the 'rob.py' file
    # 
    # we also create an optimizer

    model = SmallPFRNNModel(model_config)  
    #model = SVMParamterEstimator(model_config)
    #model = LSTM1(1, config["window_size"], 150, 1)
    if torch.cuda.is_available() and config["use_gpu"]:
        model.to('cuda')
    optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"])

    # if flag set in config we load our model parameters from a previous
    # iteration/checkpoint
    if config["load_model_from_previous"] and \
       os.path.isfile(config["model_path"]):
        # we need to figure out what device we are using so we can map the state dict
        # to that device
        current_device = torch.device("cpu")
        if torch.cuda.is_available() and config["use_gpu"]:
            current_device = torch.device("cuda")
        print("loading model from checkpoint")
        model.load_state_dict(torch.load(config["model_path"], map_location=current_device))

    # step 3: train the model
    # 
    # we perform a number of epochs.
    # at each epoch we:
    # 1. set the model to train
    # 2. train the model over successive iterations
    # 3. eval the model

    batch_size = config["batch_size"]

    loss_per_epoch = []
    training_loss = []

    for e in range(config["epochs"]):
        print(f"running epoch {e} out of {config['epochs']}")
        start_time = time.time()

        # store each iterations losses in a separate list
        training_loss.append([])

        iterations = math.ceil(len(xs_train) / config["batch_size"])
        for i in range(iterations):

            # set model to training mode
            model.train()

            # print free GPU memory (for debugging)
            # if torch.cuda.is_available():
            #     r = torch.cuda.memory_reserved(0)
            #     a = torch.cuda.memory_allocated(0)
            #     f = r-a  # free inside reserved
            #     print(f"free memory: {f}")

            xs_batch = xs_train[i * batch_size:(i + 1) * batch_size]
            ys_batch = ys_train[i * batch_size:(i + 1) * batch_size]

            # reset all gradients that have been built up on the model
            # between iterations
            optimizer.zero_grad()

            # convert tensors to cuda if available
            if torch.cuda.is_available() and config["use_gpu"]:
                xs_batch = xs_batch.to('cuda')
                ys_batch = ys_batch.to('cuda')

            # perform 1 step of gradient descent
            loss, log_loss, particle_pred = model.step(xs_batch,ys_batch,model_args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # print our loss and save it in a list
            estimated_time_remaining = int(((time.time() - start_time) / (i + 1)) * (iterations - i - 1))
            print(f"[{i + 1}/{iterations}] training loss: {loss}   [{estimated_time_remaining}s remaining]")
            training_loss[e].append(loss.to('cpu').detach().item())

        # we now evaluate the model using our eval 
        with torch.no_grad():
            model.eval()
            model.zero_grad()
            if torch.cuda.is_available() and config["use_gpu"]:
                xs_test = xs_test.to('cuda')
                ys_test = ys_test.to('cuda')
            loss, log_loss, particle_pred = model.step(xs_test, ys_test, model_args)
            print(f"eval loss: {loss}")
            loss_per_epoch.append(loss.to('cpu').detach().item())

        # save the model in between epochs
        if config["save_models"]:
            torch.save(model.state_dict(), f"{config['base_path']}_{e}.pt")
        
        print(f"epoch {e} took {time.time() - start_time} seconds")

    
    # step 4: save the model
    #
    # we save the model using pytorch functions
    torch.save(model.state_dict(), "./models/pfrnn.pt")

    
    # step 5: draw the graphs
    # 
    # we want to display our results now
    # we draw a number of graphs to show these including
    # 1. a graph of the loss over epochs
    # 2. a graph showing the model fit to data

    print(loss_per_epoch)

    # we will now predict volatility for a single innovations series
    # and then plot the predictions against the actual volatility
    def create_plot_data(series_num):

        single_series = xs_test[-series_num:(-series_num)+1]
        if torch.cuda.is_available() and config["use_gpu"]:
                    single_series = single_series.to('cuda')
        ys_pred, particle_pred = model.forward(single_series)

        # convert to numpy arrays
        ys_pred = ys_pred.cpu().detach().numpy()
        ys_true = ys_test.cpu().detach().numpy()
        particle_pred = particle_pred.cpu().detach().numpy()

        # flatten into a 1D array
        ys_pred = ys_pred.reshape((len(ys_pred), ))
        ys_true = ys_true[-series_num].reshape((len(ys_test[-series_num], )))
        xs_true = xs_test[-series_num, :,-1].reshape((len(xs_test[-series_num]), ))
        particle_pred_y = particle_pred.flatten()

        # create particle pred y
        num_particles = particle_pred.shape[1]
        sequence_length = particle_pred.shape[0]
        particle_pred_x = np.zeros(num_particles * sequence_length)
        for xi in range(0, sequence_length):
            for pi in range(0, num_particles):
                particle_pred_x[xi * num_particles + pi] = xi

        return ys_pred, ys_true, xs_true, particle_pred_x, particle_pred_y


    # plot our predicted volatility, real volatility and innovations
    # ys_pred, ys_true, xs_true = create_plot_data(7)
    # example_plot = plt.figure(1)
    # plt.plot(xs_true, color="pink",  label="innovation", linewidth=0.2)
    # plt.plot(ys_pred, color="orange", label="pred")
    # plt.plot(ys_true, color="blue", label="true")
    # example_plot.legend(loc="upper left")
    # example_plot.show()

    # # below we plot our loss graphs
    # loss_plot = plt.figure(2)
    # plt.plot(loss_per_epoch)
    # # calculate mean training loss per epoch (i.e. mean across iterations)
    # training_loss = [sum(l)/len(l) for l in training_loss]
    # plt.plot(training_loss)
    # loss_plot.show()

    # create a plot that allows us to click through different plots

    plot_innovation = False
    plot_particles = True
    const_min_lim = -3
    const_max_lim = 3

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ys_pred, ys_true, xs_true, particle_x, particle_y = create_plot_data(7)

    ys_pred_line, = plt.plot(ys_pred, color="orange", label="pred")
    ys_true_line, = plt.plot(ys_true, color="blue", label="true")
    if plot_innovation:
        xs_true_line, = plt.plot(xs_true, color="pink",  label="innovation", linewidth=0.3)
    if plot_particles:
        particle_plot = plt.scatter(particle_x, particle_y, color="lightskyblue", label="particles", s=0.04)
    plt.legend(loc="upper left")

    class Index(object):
        ind = 7
        def next(self, event):
            self.ind += 1 
            self.update()

        def prev(self, event):
            self.ind -= 1 
            self.update()

        def update(self):
            i  = self.ind %(len(xs_test))
            ys_pred,ys_true,xs_true,particle_x, particle_y = create_plot_data(i) #unpack tuple data
            y_data = list(range(len(ys_pred)))
            ys_pred_line.set_ydata(ys_pred)
            ys_pred_line.set_xdata(y_data)
            ys_true_line.set_ydata(ys_true)
            ys_true_line.set_xdata(y_data)
            if plot_innovation:
                xs_true_line.set_ydata(xs_true)
                xs_true_line.set_xdata(y_data)
            if plot_particles:
                particle_plot.set_offsets(np.column_stack((particle_x, particle_y)))
            
            y_min = min(min(ys_pred), min(ys_true))
            y_max = max(max(ys_pred), max(ys_true))
            if plot_innovation:
                y_min = min(y_min, min(xs_true))
                y_max = max(y_max, max(xs_true))
            if plot_particles:
                y_min = min(y_min, min(particle_y))
                y_max = max(y_max, max(particle_y))
            if const_min_lim:
                y_min = const_min_lim
            if const_max_lim:
                y_max = const_max_lim
            ax.set_ylim(
                y_min, 
                y_max
            )
            plt.draw()
        
    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()

    input()
    
