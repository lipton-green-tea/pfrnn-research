# Introduction
This is the code repository for my PF-RNN containing the experiments and implementation.

## Install the requirements
```
pip install -r requirements.txt
```

## Running it
You can specify a json of paramters in the ./configs directory. You can then pass it in as shown below.
```
python experiment.py ./configs/eval.json
```

It should run the experiment ending with an interactive plot that allows you to visualize the performance of the filter.

To stop running the program close the graph and press any key and then enter in the command prompt.


## Overview of important files and folders
experiment.py: the main file where experiments are run (i.e. model trained, data generated, and model evaluated)
visualisation.py: class to help visualize the particles and estimated volatility
svpfrnns.py: my various sv-pf-rnn cells used in model building
harvey_sv_model.py: the full sv-pf-rnn model that uses the sv-pf-rnn cell
particle_filter.py: abstract implementation of a particle filter
particle_filter_model.py: particle filter for volatility filtering with the same interface as our other models
pretraining.ipynb: used to pretrain the networks
real_world_data_generator.py: generates the volatility datasets in ./volatility_data
garch_pfrnn_model.py: untrained model that performs adaptive filtering based on a garch model
evaluation_helpers.py: functions to calculate different metrics
stochastic_volatility.py: holds functions for generating/simulating stochastic volatility series 

/models: holds saved checkpoints of previously trained models
/pretrained_weights: holds the weights from pretraining.ipynb
/volatility_data: pandas dataframes of volatility and returns data we created
/configs: holds the configuration files for experiments



## License and credit
This repo began as a fork of the https://github.com/Yusufma03/pfrnns/blob/master/pfrnns.py repo, and the original license has been maintained as such.