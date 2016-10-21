#
# Calibration of Gruenbichler and Longstaff (1996)
# square-root diffusion model to
# VSTOXX call options traded at Eurex
# Data as of 31. March 2014
# All data from www.eurexchange.com
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import numpy as np
import pandas as pd
from srd_functions import call_price
import scipy.optimize as sco
import matplotlib.pyplot as plt

path = 'data/'

# Fixed Parameters
v0 = 17.6639  # VSTOXX index on 31. March 2014
r = 0.01  # risk-less short rate
zeta = 0.  # volatility risk premium factor


def read_select_quotes(path=path, tol=0.2):
    ''' Selects and read options quotes.

    Parameters
    ==========
    path: string
        path to file with option quotes

    Returns
    =======
    option_data: pandas DataFrame object
        option data
    '''
    h5 = pd.HDFStore(path + 'vstoxx_march_2014.h5', 'r')

    # read option data from file and close it
    option_data = h5['vstoxx_options']
    h5.close()
    # select relevant date for call option quotes
    option_data = option_data[(option_data.DATE == '2014-3-31')
                            & (option_data.TYPE == 'C')]
    # calculate time-to-maturity in year fractions
    option_data['TTM'] = (option_data.MATURITY - option_data.DATE).apply(
                        lambda x: x / np.timedelta64(1, 'D') / 365.)

    # only those options close enough to the ATM level
    option_data = option_data[(option_data.STRIKE > (1 - tol) * v0)
                            & (option_data.STRIKE < (1 + tol) * v0)]
    return option_data


def valuation_function(p0):
    ''' Valuation function for set of strike prices

    Parameters
    ==========
    p0: list
        set of model parameters

    Returns
    =======
    call_prices: NumPy ndarray object
        array of call prices
    '''
    kappa, theta, sigma = p0
    call_prices = []
    for strike in strikes:
        call_prices.append(call_price(v0, kappa, theta,
                                   sigma, zeta, ttm, r, strike))
    call_prices = np.array(call_prices)
    return call_prices


def error_function(p0):
    ''' Error function for model calibration.

    Parameters
    ==========
    p0: tuple
        set of model parameters

    Returns
    =======
    MSE: float
        mean squared (relative/absolute) error
    '''
    global i
    call_prices = valuation_function(p0)
    kappa, theta, sigma = p0
    pen = 0.
    if 2 * kappa * theta < sigma ** 2:
        pen = 1000.0
    if kappa < 0 or theta < 0 or sigma < 0:
        pen = 1000.0
    if relative is True:
        MSE = (np.sum(((call_prices - call_quotes) / call_quotes) ** 2)
                / len(call_quotes) + pen)
    else:
        MSE = np.sum((call_prices - call_quotes) ** 2) / len(call_quotes) + pen

    if i == 0:
            print ("{:>6s} {:>6s} {:>6s}".format('kappa', 'theta', 'sigma')
                 + "{:>12s}".format('MSE'))

    # print intermediate results: every 100th iteration
    if i % 100 == 0:
        print "{:6.3f} {:6.3f} {:6.3f}".format(*p0) + "{:>12.5f}".format(MSE)
    i += 1
    return MSE


def model_calibration(option_data, rel=False, mat='2014-07-18'):
    ''' Function for global and local model calibration.

    Parameters
    ==========
    option_data: pandas DataFrame object
        option quotes to be used
    relative: boolean
        relative or absolute MSE
    maturity: string
        maturity of option quotes to calibrate to

    Returns
    =======
    opt: tuple
        optimal parameter values
    '''
    global relative  # if True: MSRE is used, if False: MSAE
    global strikes
    global call_quotes
    global ttm
    global i

    relative = rel
    # only option quotes for a single maturity
    option_quotes = option_data[option_data.MATURITY == mat]

    # time-to-maturity from the data set
    ttm = option_quotes.iloc[0, -1]

    # transform strike column and price column in ndarray object
    strikes = option_quotes['STRIKE'].values
    call_quotes = option_quotes['PRICE'].values

    # global optimization
    i = 0  # counter for calibration iterations
    p0 = sco.brute(error_function, ((5.0, 20.1, 1.0), (10., 30.1, 1.25),
                             (1.0, 9.1, 2.0)), finish=None)

    # local optimization
    i = 0
    opt = sco.fmin(error_function, p0, xtol=0.0000001, ftol=0.0000001,
                                 maxiter=1000, maxfun=1500)

    return opt


def plot_calibration_results(opt):
    ''' Function to plot market quotes vs. model prices.

    Parameters
    ==========
    opt: list
        optimal parameters from calibration
    '''
    callalues = valuation_function(opt)
    diffs = callalues - call_quotes
    plt.figure()
    plt.subplot(211)
    plt.plot(strikes, call_quotes, label='market quotes')
    plt.plot(strikes, callalues, 'ro', label='model prices')
    plt.ylabel('option values')
    plt.grid(True)
    plt.legend()
    plt.axis([min(strikes) - 0.5, max(strikes) + 0.5,
          0.0, max(call_quotes) * 1.1])
    plt.subplot(212)
    wi = 0.3
    plt.bar(strikes - wi / 2, diffs, width=wi)
    plt.grid(True)
    plt.xlabel('strike price')
    plt.ylabel('difference')
    plt.axis([min(strikes) - 0.5, max(strikes) + 0.5,
          min(diffs) * 1.1, max(diffs) * 1.1])
    plt.tight_layout()

if __name__ == '__main__':
    option_data = read_select_quotes()
    opt = model_calibration(option_data=option_data)
