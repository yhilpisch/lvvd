#
# Calibration of square-root jump diffusion (SRJD) model
# to VSTOXX European call options traded at Eurex
# Data as of 31. March 2014
# All data from www.eurexchange.com
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
from srd_model_calibration import path, read_select_quotes
from srjd_simulation import srjd_call_valuation

# fixed parameters
r = 0.01  # risk-less short rate
v0 = 17.6639  # VSTOXX index at 31.03.2014
M = 15  # number of time intervals
I = 100  # number of simulated paths


def srjd_valuation_function(p0):
    ''' Valuation ('difference') function for all options
        of a given DataFrame object.

    Parameters
    ==========
    p0: list
        set of model parameters

    Returns
    =======
    diffs: NumPy ndarray object
        array with valuation differences
    '''
    global relative, option_data
    kappa, theta, sigma, lamb, mu, delta = p0
    diffs = []
    for i, option in option_data.iterrows():
        value = srjd_call_valuation(v0, kappa, theta, sigma,
                                    lamb, mu, delta,
                                    option['TTM'], r, option['STRIKE'],
                                    M=M, I=I, fixed_seed=True)
        if relative is True:
            diffs.append((value - option['PRICE']) / option['PRICE'])
        else:
            diffs.append(value - option['PRICE'])
    diffs = np.array(diffs)
    return diffs


def srjd_error_function(p0):
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
    global i, min_MSE, option_data
    OD = len(option_data)
    diffs = srjd_valuation_function(p0)
    kappa, theta, sigma, lamb, mu, delta = p0

    # penalties
    pen = 0.
    if 2 * kappa * theta < sigma ** 2:
        pen = 1000.0
    if kappa < 0 or theta < 0 or sigma < 0 or lamb < 0 or delta < 0:
        pen = 1000.0

    MSE = np.sum(diffs ** 2) / OD + pen  # mean squared error

    min_MSE = min(min_MSE, MSE)  # running minimum value

    if i == 0:
        print '\n' + ('{:>5s}'.format('its')
                      + '{:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}'.format(
            'kappa', 'theta', 'sigma', 'lamb', 'mu', 'delta')
            + '{:>12s}'.format('MSE') + '{:>12s}'.format('min_MSE'))
    # print intermediate results: every 100th iteration
    if i % 100 == 0:
        print ('{:>5d}'.format(i)
               + '{:7.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format(*p0)
               + '{:>12.5f}'.format(MSE) + '{:>12.5f}'.format(min_MSE))
    i += 1
    return MSE


def srjd_model_calibration(data, p0=None, rel=False, mats=None):
    ''' Function for global and local model calibration.

    Parameters
    ==========
    option_data: pandas DataFrame object
        option quotes to be used
    relative: bool
        relative or absolute MSE
    mats: list
        list of maturities of option quotes to calibrate to

    Returns
    =======
    opt: tuple
        optimal parameter values
    '''
    global i, min_MSE, option_data
    global relative  # if True: MSRE is used, if False: MSAE

    min_MSE = 5000.  # dummy value
    relative = rel  # relative or absolute
    option_data = data

    if mats is not None:
        # select the option data for the given maturities
        option_data = option_data[option_data['MATURITY'].isin(mats)]

    # global optimization
    if p0 is None:
        i = 0  # counter for calibration iterations
        p0 = sco.brute(srjd_error_function, (
            (1.0, 9.1, 4.0),  # kappa
            (10., 20.1, 10.0),  # theta
            (1.0, 3.1, 2.0),  # sigma
            (0.0, 0.81, 0.4),  # lambda
            (-0.2, 0.41, 0.3),  # mu
            (0.0, 0.31, 0.15)),  # delta
            finish=None)

    # local optimization
    i = 0
    opt = sco.fmin(srjd_error_function, p0,
                   xtol=0.0000001, ftol=0.0000001,
                   maxiter=550, maxfun=700)

    return opt


def plot_calibration_results(option_data, opt, mats):
    ''' Function to plot market quotes vs. model prices.

    Parameters
    ==========
    option_data: pandas DataFrame object
        option data to plot
    opt: list
        optimal results from calibration
    mats: list
        maturities to be plotted
    '''
    kappa, theta, sigma, lamb, mu, delta = opt
    # adding model values for optimal parameter set
    # to the DataFrame object
    values = []
    for i, option in option_data.iterrows():
        value = srjd_call_valuation(v0, kappa, theta, sigma,
                                    lamb, mu, delta,
                                    option['TTM'], r, option['STRIKE'],
                                    M=M, I=I, fixed_seed=True)
        values.append(value)
    option_data['MODEL'] = values

    # plotting the market and model values
    height = min(len(mats) * 3, 12)
    fig, axarr = plt.subplots(len(mats), 2, sharex=True, figsize=(10, height))
    for z, mat in enumerate(mats):
        if z == 0:
            axarr[z, 0].set_title('values')
            axarr[z, 1].set_title('differences')
        os = option_data[option_data.MATURITY == mat]
        strikes = os.STRIKE.values
        axarr[z, 0].set_ylabel('%s' % str(mat)[:10])
        axarr[z, 0].plot(strikes, os.PRICE.values, label='market quotes')
        axarr[z, 0].plot(strikes, os.MODEL.values, 'ro', label='model prices')
        axarr[z, 0].legend(loc=0)
        wi = 0.3
        axarr[z, 1].bar(strikes - wi / 2, os.MODEL.values - os.PRICE.values,
                        width=wi)
        if mat == mats[-1]:
            axarr[z, 0].set_xlabel('strike')
            axarr[z, 1].set_xlabel('strike')

if __name__ == '__main__':
    option_data = read_select_quotes('./source/data/', tol=0.1)
    option_data['VALUE'] = 0.0
    opt = srjd_model_calibration()
