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