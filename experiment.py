import evaluation_helpers
from stochastic_volatility import HarveySV, HarveySVParamters
#from rob import SVMParamterEstimator, ModelArgs
from model_args import ModelArgs
from visualisation import InteractivePlot
from harvey_sv_model import HarveySVPF
from small_pfrnn_model import SmallPFRNNModel
from lstm_model import LSTM1

import os
import sys
import math
import torch
import time
import json
import numpy as np


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
    # first lets load our config
    # we check if the user has entered a desired config, otherwise we use the eval config

    if len(sys.argv) > 1:
        fp = sys.argv[1]
    else:
        fp = "./configs/eval.json"
    with open(fp, "r") as config_file:
        config = json.load(config_file)

    
    sv_parameters = HarveySVParamters(
        mu=config["sv_config"]["mu"], 
        phi=config["sv_config"]["phi"], 
        tau=config["sv_config"]["tau"]
    )

    # lets init our model arguments
    model_config = config["model_config"]
    model_args = ModelArgs(
        l1_weight=model_config["l1_weight"],
        l2_weight=model_config["l2_weight"],
        elbo_weight=model_config["elbo_weight"],
        resamp_alpha=model_config["resamp_alpha"]
    )

    # here we either load or generate our dataset
    if config["load_data_from_previous"] and \
       os.path.isfile("./simulated_data/xs_train.pt"):  # if the xs_train tensor file exists assume the others do too
        # we load our tensors from their files
        with open("./simulated_data/xs_train.pt", 'rb') as f:
            xs_train = torch.load(f)
        with open("./simulated_data/ys_train.pt", 'rb') as f:
            ys_train = torch.load(f)
        with open("./simulated_data/xs_test.pt", 'rb') as f:
            xs_test = torch.load(f)
        with open("./simulated_data/ys_test.pt", 'rb') as f:
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
        with open("./simulated_data/xs_train.pt", 'wb') as f:
            torch.save(xs_train, f)
        with open("./simulated_data/ys_train.pt", 'wb') as f:
            torch.save(ys_train, f)
        with open("./simulated_data/xs_test.pt", 'wb') as f:
            torch.save(xs_test, f)
        with open("./simulated_data/ys_test.pt", 'wb') as f:
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
            loss, mean_mse_loss, particle_pred = model.step(xs_batch,ys_batch,model_args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # print our loss and save it in a list
            estimated_time_remaining = int(((time.time() - start_time) / (i + 1)) * (iterations - i - 1))
            print(f"[{i + 1}/{iterations}] training loss (loss func -- mean mse): {loss} --  {mean_mse_loss}  [{estimated_time_remaining}s remaining]")
            training_loss[e].append(loss.to('cpu').detach().item())

        # we now evaluate the model using our eval 
        with torch.no_grad():
            model.eval()
            model.zero_grad()
            if torch.cuda.is_available() and config["use_gpu"]:
                xs_test = xs_test.to('cuda')
                ys_test = ys_test.to('cuda')
            loss, mean_mse_loss, particle_pred = model.step(xs_test, ys_test, model_args)
            print(f"eval loss: {loss}")
            print(f"eval mean mse loss: {mean_mse_loss}")
            loss_per_epoch.append(loss.to('cpu').detach().item())

        # save the model in between epochs
        if config["save_models"]:
            torch.save(model.state_dict(), f"{config['base_path']}_{e}.pt")
        
        print(f"epoch {e} took {time.time() - start_time} seconds")


    # lets print out our final stats
    print(f"eval loss: {loss_per_epoch}")
    training_loss = [sum(iter_loss)/len(iter_loss) for iter_loss in training_loss]
    print(f"training loss: {training_loss}")

    
    # step 4: save the model and loss stats
    #
    # we save the model using pytorch functions
    if config["epochs"] > 0:
        torch.save(model.state_dict(), f"{config['base_path']}_final")

    # write the loss to a file
    loss_file = open('./saved_loss/loss.txt','w')
    loss_file.write(str(loss_per_epoch)+"\n")
    loss_file.write(str(training_loss)+"\n")
    loss_file.close()


    # step 5: calculate our metrics/statistics
    # we calculate our 5 main statistics over the entire batch of test data
    if torch.cuda.is_available() and config["use_gpu"]:
        xs_test = xs_test.to('cuda')
    ys_pred, particle_pred = model.forward(xs_test)

    ys_pred = ys_pred.cpu().detach().numpy()
    ys_true = ys_test.cpu().detach().numpy()
    particle_pred = particle_pred.cpu().detach().numpy()
    
    test_results = {
        "mse": np.zeros(len(ys_test)),
        "mae": np.zeros(len(ys_test)),
        "qlike": np.zeros(len(ys_test)),
        "mde": np.zeros(len(ys_test)),
        "log_likelihood": np.zeros(len(ys_test)),
        "particle_log_likelihood": np.zeros(len(ys_test)),
    } 
    real = np.squeeze(ys_true)
    pred = np.squeeze(ys_pred.transpose((1,0,2)))
    # we need to reshape the particles into a useable shape
    reshaped_particles = particle_pred.reshape((config["sequence_length"],len(ys_true),model_config["num_particles"])).transpose((1,0,2))
    test_results["mse"] = evaluation_helpers.mse(real, pred)
    test_results["mae"] = evaluation_helpers.mae(real, pred)
    test_results["qlike"] = evaluation_helpers.qlike(real, pred)
    test_results["mde"] = evaluation_helpers.mde(real, pred)
    test_results["log_likelihood"] = evaluation_helpers.log_likelihood(real, pred, sv_parameters.tau)

    for k, v in test_results.items():
        print(f"mean {k}: {v.mean()}")
        print(f"std  {k}: {v.std()}")

    
    # step 5: draw the graphs
    # 
    # we want to display our results now
    # we draw a number of graphs to show these including
    # 1. a graph of the loss over epochs
    # 2. a graph showing the model fit to data

    # # below we plot our loss graphs
    # loss_plot = plt.figure(2)
    # plt.plot(loss_per_epoch)
    # # calculate mean training loss per epoch (i.e. mean across iterations)
    # training_loss = [sum(l)/len(l) for l in training_loss]
    # plt.plot(training_loss)
    # loss_plot.show() 


    # create an interactive plot that allows us to explore how our model fits to data
    plot_config = config["plot_config"]
    iplot = InteractivePlot(model, xs_test, ys_test, config=plot_config)
    iplot.init_plot()

    input()
    
