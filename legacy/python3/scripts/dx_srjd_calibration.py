#
# Calibration of Square-Root Jump Diffusion (SRJD)
# model to VSTOXX call options with DX Analytics
#
# All data from www.eurexchange.com
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import dx
import time
import numpy as np
import pandas as pd
import datetime as dt
import scipy.optimize as spo
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
from copy import deepcopy

# importing the data
h5 = pd.HDFStore('./data/vstoxx_march_2014.h5', 'r')
vstoxx_index = h5['vstoxx_index']
vstoxx_futures = h5['vstoxx_futures']
vstoxx_options = h5['vstoxx_options']
h5.close()
h5.close()
vstoxx_futures['DATE'] = pd.to_datetime(vstoxx_futures['DATE'])
vstoxx_futures['MATURITY'] = pd.to_datetime(vstoxx_futures['MATURITY'])
vstoxx_options['DATE'] = pd.to_datetime(vstoxx_options['DATE'])
vstoxx_options['MATURITY'] = pd.to_datetime(vstoxx_options['MATURITY'])
# collecting the maturity dates
third_fridays = sorted(set(vstoxx_futures['MATURITY']))

# instantiation of market environment object with dummy pricing date
me_vstoxx = dx.market_environment('me_vstoxx', dt.datetime(2014, 1, 1))
me_vstoxx.add_constant('currency', 'EUR')
me_vstoxx.add_constant('frequency', 'W')
me_vstoxx.add_constant('paths', 5000)

# constant short rate model with somewhat arbitrary rate
csr = dx.constant_short_rate('csr', 0.01)
me_vstoxx.add_curve('discount_curve', csr)

# parameters to be calibrated later, dummies only
# SRD part
me_vstoxx.add_constant('kappa', 1.0)
me_vstoxx.add_constant('theta', 20)
me_vstoxx.add_constant('volatility', 1.0)
# jump part
me_vstoxx.add_constant('lambda', 0.5)
me_vstoxx.add_constant('mu', -0.2)
me_vstoxx.add_constant('delta', 0.1)

# payoff function for all European call options
payoff_func = 'np.maximum(maturity_value - strike, 0)'

tol = 0.2  # otm & itm moneyness tolerance
first = True  # flag for first calibration

def srjd_get_option_selection(pricing_date, tol=tol):
    ''' Function to select option quotes from data set.

    Parameters
    ==========
    pricing_date: datetime object
        date for which the calibration shall be implemented
    tol: float
        moneyness tolerace for OTM and ITM options to be selected

    Returns
    =======
    option_selection: DataFrame object
        selected options quotes
    futures: DataFrame object
        futures prices at pricing_date
    '''
    global mn
    option_selection = pd.DataFrame()
    mats = [third_fridays[3]]  # list of single maturity date
    # mats = third_fridays[3:5]  # list of two maturity dates
    # mats = third_fridays[3:8]  # list of five maturity dates
    print(mats)
    mn = len(mats)
    # select the relevant futures prices
    futures = vstoxx_futures[(vstoxx_futures.DATE == pricing_date)
            &  (vstoxx_futures.MATURITY.apply(lambda x: x in mats))]
    # collect option data for the given option maturities
    for mat in mats:
        forward = futures[futures.MATURITY == mat]['PRICE'].values[0]
        option_selection = option_selection.append(
            vstoxx_options[(vstoxx_options.DATE == pricing_date)
                         & (vstoxx_options.MATURITY == mat)
                         & (vstoxx_options.TYPE == 'C')  # only calls
                         & (vstoxx_options.STRIKE > (1 - tol) * forward)
                         & (vstoxx_options.STRIKE < (1 + tol) * forward)])
    return option_selection, futures


def srd_forward_error(p0):
    ''' Calculates the mean-squared error for the
    term structure calibration for the SRD model part.

    Parameters
    ===========
    p0: tuple/list
        tuple of kappa, theta, volatility

    Returns
    =======
    MSE: float
        mean-squared error
    '''
    global initial_value, f, t
    if p0[0] < 0 or p0[1] < 0 or p0[2] < 0:
        return 100
    f_model = dx.srd_forwards(initial_value, p0, t)
    MSE = np.sum((f - f_model) ** 2) / len(f)
    return MSE


def generate_shift_base(pricing_date, futures):
    ''' Generates the values for the deterministic shift for the
    SRD model part.

    Parameters
    ==========
    pricing_date: datetime object
        date for which the calibration shall be implemented
    futures: DataFrame object
        futures prices at pricing_date

    Returns
    =======
    shift_base: ndarray object
        shift values for the SRD model part
    '''
    global initial_value, f, t
    # futures price array
    f = list(futures['PRICE'].values)
    f.insert(0, initial_value)
    f = np.array(f)
    # date array
    t = [_.to_pydatetime() for _ in futures['MATURITY']]
    t.insert(0, pricing_date)
    t = np.array(t)
    # calibration to the futures values
    opt = spo.fmin(srd_forward_error, (2., 15., 2.))
    # calculation of shift values
    f_model = dx.srd_forwards(initial_value, opt, t)
    shifts = f - f_model
    shift_base = np.array((t, shifts)).T
    return shift_base


def srjd_get_option_models(pricing_date, option_selection, futures):
    ''' Function to instantiate option pricing models.

    Parameters
    ==========
    pricing_date: datetime object
        date for which the calibration shall be implemented
    maturity: datetime object
        maturity date for the options to be selected
    option_selection: DataFrame object
        selected options quotes

    Returns
    =======
    vstoxx_model: dx.square_root_diffusion
        model object for VSTOXX
    option_models: dict
        dictionary of dx.valuation_mcs_european_single objects
    '''
    global initial_value
    # updating the pricing date
    me_vstoxx.pricing_date = pricing_date
    # setting the initial value for the pricing date
    initial_value = vstoxx_index['V2TX'][pricing_date]
    me_vstoxx.add_constant('initial_value', initial_value)
    # setting the final date given the maturity dates
    final_date = max(futures.MATURITY).to_pydatetime()
    me_vstoxx.add_constant('final_date', final_date)
    # adding the futures term structure
    me_vstoxx.add_curve('term_structure', futures)
    # instantiating the risk factor (VSTOXX) model
    vstoxx_model = dx.square_root_jump_diffusion_plus('vstoxx_model',
                                                      me_vstoxx)
    # generating the shift values and updating the model
    vstoxx_model.shift_base = generate_shift_base(pricing_date, futures)
    vstoxx_model.update_shift_values()

    option_models = {}  # dictionary collecting all models
    for option in option_selection.index:
        # setting the maturity date for the given option
        maturity = option_selection['MATURITY'].loc[option]
        me_vstoxx.add_constant('maturity', maturity)
        # setting the strike for the option to be modeled
        strike = option_selection['STRIKE'].loc[option]
        me_vstoxx.add_constant('strike', strike)
        # instantiating the option model
        option_models[option] = \
                            dx.valuation_mcs_european_single(
                                    'eur_call_%d' % strike,
                                    vstoxx_model,
                                    me_vstoxx,
                                    payoff_func)
    return vstoxx_model, option_models


def srjd_calculate_model_values(p0):
    ''' Returns all relevant option values.

    Parameters
    ===========
    p0: tuple/list
        tuple of kappa, theta, volatility, lamb, mu, delt

    Returns
    =======
    model_values: dict
        dictionary with model values
    '''
    # read the model parameters from input tuple
    kappa, theta, volatility, lamb, mu, delt = p0
    # update the option market environment
    vstoxx_model.update(kappa=kappa,
                        theta=theta,
                        volatility=volatility,
                        lamb=lamb,
                        mu=mu,
                        delt=delt)
    # estimate and collect all option model present values
    results = [option_models[option].present_value(fixed_seed=True)
               for option in option_models]
    # combine the results with the option models in a dictionary
    model_values = dict(zip(option_models, results))
    return model_values


def srjd_mean_squared_error(p0, penalty=True):
    ''' Returns the mean-squared error given
    the model and market values.

    Parameters
    ===========
    p0: tuple/list
        tuple of kappa, theta, volatility

    Returns
    =======
    MSE: float
        mean-squared error
    '''
    # escape with high value for non-sensible parameter values
    if (p0[0] < 0 or p0[1] < 5. or p0[2] < 0 or p0[2] > 10.
        or p0[3] < 0 or p0[4] < 0 or p0[5] < 0):
        return 1000
    # define/access global variables/objects
    global option_selection, vstoxx_model, option_models, first, last
    # calculate the model values for the option selection
    model_values = srjd_calculate_model_values(p0)
    option_diffs = {}  # dictionary to collect differences
    for option in model_values:
        # differences between model value and market quote
        option_diffs[option] = (model_values[option]
                             - option_selection['PRICE'].loc[option])
    # calculation of mean-squared error
    MSE = np.sum(np.array(list(option_diffs.values())) ** 2) / len(option_diffs)
    if first:
        # if in first optimization, no penalty
        pen = 0.0
    else:
        # if 2, 3, ... optimization, penalize deviation from previous
        # optimal parameter combination
        pen = np.mean((p0 - last) ** 2)
    if not penalty:
        return MSE
    return MSE + pen


def srjd_get_parameter_series(pricing_date_list):
    ''' Returns parameter series for the calibrated model over time.

    Parameters
    ==========
    pricing_date_list: pd.DatetimeIndex
        object with relevant pricing dates

    Returns
    =======
    parameters: pd.DataFrame
        DataFrame object with parameter series
    '''
    # define/access global variables/objects
    global initial_value, futures, option_selection, vstoxx_model, \
            option_models, first, last
    parameters = pd.DataFrame()  # DataFrame object to collect parameter series
    for pricing_date in pricing_date_list:
        # setting the initial value for the VSTOXX
        initial_value = vstoxx_index['V2TX'][pricing_date]
        # select relevant option quotes
        option_selection, futures = srjd_get_option_selection(pricing_date)
        # instantiate all model given option selection
        vstoxx_model, option_models = srjd_get_option_models(pricing_date,
                                                        option_selection,
                                                        futures)
        if first:    
            # global optimization to start with
            opt = spo.brute(srjd_mean_squared_error,
                ((1.25, 6.51, 0.75),   # range for kappa
                 (10., 20.1, 2.5),   # range for theta
                 (0.5, 10.51, 2.5),  # range for volatility
                 (0.1, 0.71, 0.3),  # range for lambda
                 (0.1, 0.71, 0.3),  # range for mu
                 (0.1, 0.21, 0.1)),  # range for delta
                 finish=None)
        # local optimization
        opt = spo.fmin(srjd_mean_squared_error, opt,
                       maxiter=550, maxfun=650,
                       xtol=0.0000001, ftol=0.0000001);
        # calculate MSE for storage
        MSE = srjd_mean_squared_error(opt)
        # store main parameters and results
        parameters = parameters.append(
                 pd.DataFrame(
                 {'date' : pricing_date,
                  'initial_value' : vstoxx_model.initial_value,
                  'kappa' : opt[0],
                  'theta' : opt[1],
                  'sigma' : opt[2],
                  'lambda' : opt[3],
                  'mu' : opt[4],
                  'delta' : opt[5],
                  'MSE' : MSE},
                  index=[0]), ignore_index=True)
        first = False  # set to False after first iteration
        last = opt  # store optimal parameters for reference
        print ('Pricing Date %s' % str(pricing_date)[:10]
               + ' | MSE %6.5f' % MSE)
    return parameters


def srjd_plot_model_fit(parameters):
    # last pricing date
    pdate = max(parameters.date)
    # optimal parameters for that date and the maturity
    opt = np.array(parameters[parameters.date == pdate][[
        'kappa', 'theta', 'sigma', 'lambda', 'mu', 'delta']])[0]
    option_selection, futures = srjd_get_option_selection(pdate, tol=tol)
    vstoxx_model, option_models = srjd_get_option_models(pdate,
                                                    option_selection,
                                                    futures)
    model_values = srjd_calculate_model_values(opt)
    model_values = pd.DataFrame(model_values.values(),
                                index=model_values.keys(),
                                columns=['MODEL'])
    option_selection = option_selection.join(model_values)
    mats = set(option_selection.MATURITY.values)
    mats = sorted(mats)
    # arranging the canvas for the subplots
    height = max(8, 2 * len(mats))
    if len(mats) == 1:
        mat = mats[0]
        fig, axarr = plt.subplots(2, figsize=(10, height))
        os = option_selection[option_selection.MATURITY == mat]
        strikes = os.STRIKE.values
        axarr[0].set_ylabel('%s' % str(mat)[:10])
        axarr[0].plot(strikes, os.PRICE.values, label='Market Quotes')
        axarr[0].plot(strikes, os.MODEL.values, 'ro', label='Model Prices')
        axarr[0].legend(loc=0)
        wi = 0.3
        axarr[1].bar(strikes, os.MODEL.values - os.PRICE.values,
                    width=wi)
        axarr[0].set_xlabel('strike')
        axarr[1].set_xlabel('strike')
    else:
        fig, axarr = plt.subplots(len(mats), 2, sharex=True, figsize=(10, height))
        for z, mat in enumerate(mats):
            os = option_selection[option_selection.MATURITY == mat]
            strikes = os.STRIKE.values
            axarr[z, 0].set_ylabel('%s' % str(mat)[:10])
            axarr[z, 0].plot(strikes, os.PRICE.values, label='Market Quotes')
            axarr[z, 0].plot(strikes, os.MODEL.values, 'ro', label='Model Prices')
            axarr[z, 0].legend(loc=0)
            wi = 0.3
            axarr[z, 1].bar(strikes, os.MODEL.values - os.PRICE.values, width=wi)
        axarr[z, 0].set_xlabel('strike')
        axarr[z, 1].set_xlabel('strike')
    plt.tight_layout()
    plt.savefig('./images/dx_srjd_cali_%d_fit.png' % mn)


if __name__ == '__main__':
    t0 = time.time()
    # selecting the dates for the calibration
    pricing_date_list = pd.date_range('2014/3/1', '2014/3/31', freq='B')
    # conducting the calibration
    parameters = srjd_get_parameter_series(pricing_date_list)
    # storing the calibation results
    date = str(dt.datetime.now())[:10]
    h5 = pd.HDFStore('../data/srjd_calibration_%s_%s_%s' %
                (me_vstoxx.get_constant('paths'),
                 me_vstoxx.get_constant('frequency'),
                 date.replace('-', '_')), 'w')
    h5['parameters'] = parameters
    h5.close()
    # plotting the parameter time series data
    # fig1, ax1 = plt.subplots(1, figsize=(10, 12))
    to_plot = parameters.set_index('date')[
                    ['kappa', 'theta', 'sigma',
                     'lambda', 'mu', 'delta', 'MSE']]
    to_plot.plot(subplots=True, color='b', title='SRJD', figsize=(10, 12))
    plt.tight_layout()
    plt.savefig('./images/dx_srjd_cali_%d.png' % mn)
    # plotting the histogram of the MSE values
    fig, ax = plt.subplots()
    dat = parameters.MSE
    dat.hist(bins=30, ax=ax)
    plt.axvline(dat.mean(), color='r', ls='dashed',
                    lw=1.5, label='mean = %5.4f' % dat.mean())
    plt.legend()
    plt.savefig('./images/dx_srjd_cali_%d_hist.png' % mn)
    # plotting the model fit at last pricing date
    srjd_plot_model_fit(parameters)
    # measuring and printing the time needed for the script execution
    print('Time in minutes %.2f' % ((time.time() - t0) / 60))
