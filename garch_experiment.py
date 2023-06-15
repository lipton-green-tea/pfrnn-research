import torch
from garch_pfrnn_model import GarchFilter
from model_args import ModelArgs
import numpy as np
import pandas as pd
from visualisation import InteractivePlot


if __name__=="__main__":
    config = {
        "window_size": 2,
        "sequence_length": 300
    }
    model_config = {
        "num_particles": 128,
        "input_size": 2,
        "hidden_dimension": 5,
        "l1_weight": 0.1,
        "l2_weight": 0.9,
        "elbo_weight": 0.5,
        "resamp_alpha": 0.35,
        "garch_paramters": {
            "const": 0,
            "q1": 0,
            "q2": 0,
            "p1": 0,
        }
    }
    model_args = ModelArgs(
        l1_weight=0,
        l2_weight=1,
        elbo_weight=0.4,
        resamp_alpha=0.3
    )
    model = GarchFilter(model_config)
    

    # lets load a dataset
    dataset = pd.read_csv("./volatility_data/GS.csv_20_year_daily_vol.csv")

    # we store our samples here
    xs = []
    ys = []

    # convert to numpy arrays
    volatility = dataset["volatility"].to_numpy()
    innovations = dataset["returns"].to_numpy()

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

    # convert it all to the right datatypes
    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    xs = torch.from_numpy(xs).tile((100,1,1))
    ys = torch.from_numpy(ys).tile((100,1,1))
    xs.type(torch.FloatTensor)
    ys.type(torch.FloatTensor)

    # create an interactive plot that allows us to explore how our model fits to data
    plot_config = {
        "use_gpu": True,
        "plot_innovations": False,
        "plot_particles": False,
        "const_min_lim": None,
        "const_max_lim": None
    }
    iplot = InteractivePlot(model, xs, ys, config=plot_config)
    iplot.init_plot()

    input()

