#
# Script for term structure calibration of
# Square-Root Jump Diffusion (SRJD) model
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import math
import numpy as np
import pandas as pd
import scipy.optimize as sco

v0 = 17.6639  # initial VSTOXX index level
i = 0  # counter for calibration runs

# reading the VSTOXX futures quotes
path = './data/'
h5 = pd.HDFStore(path + 'vstoxx_data_31032014.h5', 'r')
futures_quotes = h5['futures_data']
h5.close()

# selecting needed data columns and adding spot value
forwards = list(futures_quotes['PRICE'].values)
forwards.insert(0, v0)
forwards = np.array(forwards)
ttms = list(futures_quotes['TTM'].values)
ttms.insert(0, 0)
ttms = np.array(ttms)


def srd_forwards(p0):
    ''' Function for forward volatilities in GL96 Model.

    Parameters
    ==========
    p0: list
        set of model parameters, where

        kappa: float
            mean-reversion factor
        theta: float
            long-run mean
        sigma: float
            volatility factor

    Returns
    =======
    forwards: NumPy ndarray object
        forward volatilities
    '''
    t = ttms
    kappa, theta, sigma = p0
    g = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    sum1 = ((kappa * theta * (np.exp(g * t) - 1)) /
            (2 * g + (kappa + g) * (np.exp(g * t) - 1)))
    sum2 = v0 * ((4 * g ** 2 * np.exp(g * t)) /
                 (2 * g + (kappa + g) * (np.exp(g * t) - 1)) ** 2)
    forwards = sum1 + sum2
    return forwards


def srd_fwd_error(p0):
    ''' Error function for GL96 forward volatilities calibration.

    Parameters
    ==========
    p0: tuple
        parameter vector

    Returns
    =======
    MSE: float
        mean-squared error for p0
    '''
    global i
    kappa, theta, sigma = p0
    srd_fwds = srd_forwards(p0)
    MSE = np.sum((forwards - srd_fwds) ** 2) / len(forwards)
    if 2 * kappa * theta < sigma ** 2:
        MSE = MSE + 100   # penalty
    elif sigma < 0:
        MSE = MSE + 100
    # print intermediate results: every 50th iteration
    if i % 50 == 0:
        print("{:6.3f} {:6.3f} {:6.3f}".format(*p0) + "{:>12.5f}".format(MSE))
    i += 1
    return MSE

if __name__ == '__main__':
    p0 = 1.0, 17.5, 1.0
    opt = sco.fmin(srd_fwd_error, p0,
                   xtol=0.00001, ftol=0.00001,
                   maxiter=1500, maxfun=2000)
