#
# Module with functions for
# Variance Swaps Examples
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import math
import numpy as np
import pandas as pd

def generate_path(S0, r, sigma, T, M, seed=100000):
    ''' Function to simulate a geometric Brownian motion.

    Parameters
    ==========
    S0: float
        initial index level
    r: float
        constant risk-less short rate
    sigma: float
        instantaneous volatility
    T: float
        date of maturity (in year fractions)
    M: int
        number of time intervals

    Returns
    =======
    path: pandas DataFrame object
        simulated path
    '''
    # length of time interval
    dt = float(T) / M
    # random numbers
    np.random.seed(seed)
    rn = np.random.standard_normal(M + 1)
    rn[0] = 0  # to keep the initial value
    # simulation of path
    path = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
                                  + sigma * math.sqrt(dt) * rn))
    # setting initial value
    path = pd.DataFrame(path, columns=['index'])
    return path
