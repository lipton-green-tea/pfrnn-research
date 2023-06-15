import math
import numpy as np
import pyflux as pf
import pandas as pd
from os import listdir
from os.path import isfile, join


# in this file we replicate what we did in real_world_data.ipynb but for multiple data sources
# [!] to run this file you must be running an interpreter with python 3.9 and ensure you have intalled pyflux


# we take in a pandas dataframe of daily price data (from yahoo finance)
# and return a dataframe that contains the columns [date, returns, volatility]
def generate_vol_from_price_data(price_data):
    price_data["log_returns"] = np.log(price_data["Close"]).diff()
    returns = price_data["log_returns"].to_numpy()[1:]
    garch_model = pf.GARCH(returns, p=2, q=1)
    result = garch_model.fit()
    parameters = garch_model.transform_z()

    gp = garch_pred_generator(
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        0# parameters[4]
    )
    returns = returns - parameters[-1]  # subtract the mean of the returns
    for value in parameters:
        print("{:{width}.4f}".format(value, width=6), end=" ")
    print()
    conditional_vol = np.zeros_like(returns)
    conditional_vol[0] = price_data["log_returns"].to_numpy()[1:].std()
    conditional_vol[1] = price_data["log_returns"].to_numpy()[1:].std()

    for i in range(1, len(returns) - 1):
        conditional_vol[i + 1] = gp(returns, conditional_vol, i)
    
    volatility_data = pd.DataFrame({
        "date": price_data["Date"].to_numpy()[1:],  # get rid of the first date as it has NaN data
        "returns": returns,
        "volatility": conditional_vol
    })  

    return volatility_data
    

# generates the garch_pred function for a specific set of params
def garch_pred_generator(const_term, q1, q2, p1, p2):
    # given a dataframe of returns, and previous conditional volatility
    # along with a current index, returns the next step volatility
    def garch_pred(returns, conditional_vol, i):
        r1 = returns[i]
        r2 = returns[i-1]
        cv1 = conditional_vol[i]
        cv2 = conditional_vol[i-1]
        return math.sqrt(const_term + q1*r1**2 + q2*r2**2 + p1*cv1**2 + p2*cv2**2)
    return garch_pred


# we find all data files and create volatility data for them
if __name__=="__main__":
    data_dir = "./yahoo_finance_data"
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    for f in files:
        daily_data = pd.read_csv(f"./yahoo_finance_data/{f}")
        vol_df = generate_vol_from_price_data(daily_data)
        vol_df.to_csv(f"./volatility_data/{f}_20_year_daily_vol.csv")
    
