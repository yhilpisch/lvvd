#
# Calibration of Gruenbichler and Longstaff (1996)
# Square-Root Diffusion (SRD) model to
# VSTOXX call options with DX Analytics
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
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

# importing the data
h5 = pd.HDFStore('./data/vstoxx_march_2014.h5', 'r')
vstoxx_index = h5['vstoxx_index']
vstoxx_futures = h5['vstoxx_futures']
vstoxx_options = h5['vstoxx_options']
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
me_vstoxx.add_constant('kappa', 1.0)
me_vstoxx.add_constant('theta', 20)
me_vstoxx.add_constant('volatility', 1.0)

# payoff function for all European call options
payoff_func = 'np.maximum(maturity_value - strike, 0)'

tol = 0.2  # otm & itm moneyness tolerance

def srd_get_option_selection(pricing_date, maturity, tol=tol):
    ''' Function to select option quotes from data set.

    Parameters
    ==========
    pricing_date : datetime object
        date for which the calibration shall be implemented
    maturity: datetime object
        maturity date for the options to be selected
    tol: float
        moneyness tolerace for OTM and ITM options to be selected

    Returns
    =======
    option_selection : DataFrame object
        selected options quotes
    forward : float
        futures price for maturity at pricing_date
    '''
    forward = vstoxx_futures[(vstoxx_futures.DATE == pricing_date)
                & (vstoxx_futures.MATURITY == maturity)]['PRICE'].values[0]
    option_selection = \
        vstoxx_options[(vstoxx_options.DATE == pricing_date)
                     & (vstoxx_options.MATURITY == maturity)
                     & (vstoxx_options.TYPE == 'C')  # only calls
                     & (vstoxx_options.STRIKE > (1 - tol) * forward)
                     & (vstoxx_options.STRIKE < (1 + tol) * forward)]
    return option_selection, forward


def srd_get_option_models(pricing_date, maturity, option_selection):
    ''' Function to instantiate option pricing models.

    Parameters
    ==========
    pricing_date : datetime object
        date for which the calibration shall be implemented
    maturity : datetime object
        maturity date for the options to be selected
    option_selection : DataFrame object
        selected options quotes

    Returns
    =======
    vstoxx_model: dx.square_root_diffusion
        model object for VSTOXX
    option_models: dict
        dictionary of dx.valuation_mcs_european_single objects
    '''
    # updating the pricing date
    me_vstoxx.pricing_date = pricing_date
    # setting the initial value for the pricing date
    initial_value = vstoxx_index['V2TX'][pricing_date]
    me_vstoxx.add_constant('initial_value', initial_value)
    # setting the final date given the maturity date
    me_vstoxx.add_constant('final_date', maturity)
    # instantiating the risk factor (VSTOXX) model
    vstoxx_model = dx.square_root_diffusion('vstoxx_model', me_vstoxx)
    # setting the maturity date for the valuation model(s)
    me_vstoxx.add_constant('maturity', maturity)

    option_models = {}  # dictionary collecting all models
    for option in option_selection.index:
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


def srd_calculate_model_values(p0):
    ''' Returns all relevant option values.

    Parameters
    ===========
    p0 : tuple/list
        tuple of kappa, theta, volatility

    Returns
    =======
    model_values: dict
        dictionary with model values
    '''
    # read the model parameters from input tuple
    kappa, theta, volatility = p0
    # update the option market environment
    vstoxx_model.update(kappa=kappa,
                        theta=theta,
                        volatility=volatility)
    # estimate and collect all option model present values
    results = [option_models[option].present_value(fixed_seed=True)
               for option in option_models]
    # combine the results with the option models in a dictionary
    model_values = dict(zip(option_models, results))
    return model_values


def srd_mean_squared_error(p0, penalty=True):
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
    if p0[0] < 0 or p0[1] < 5. or p0[2] < 0 or p0[2] > 10.:
        return 1000
    # define/access global variables/objects
    global option_selection, vstoxx_model, option_models, first, last
    # calculate the model values for the option selection
    model_values = srd_calculate_model_values(p0)
    option_diffs = {}  # dictionary to collect differences
    for option in model_values:
        # differences between model value and market quote
        option_diffs[option] = (model_values[option]
                             - option_selection['PRICE'].loc[option])
    # calculation of mean-squared error
    MSE = np.sum(np.array(list(option_diffs.values())) ** 2) / len(option_diffs)
    if first:
        # if in global optimization, no penalty
        pen = 0.0
    else:
        # if in local optimization, penalize deviation from previous
        # optimal parameter combination
        pen = np.mean((p0 - last) ** 2)
    if not penalty:
        return MSE
    return MSE + pen


def srd_get_parameter_series(pricing_date_list, maturity_list):
    ''' Returns parameter series for the calibrated model over time.

    Parameters
    ==========
    pricing_date_list: pd.DatetimeIndex
        object with relevant pricing dates
    maturity_list: list
        list with maturities to be calibrated

    Returns
    =======
    parameters: pd.DataFrame
        DataFrame object with parameter series
    '''
    # define/access global variables/objects
    global option_selection, vstoxx_model, option_models, first, last
    parameters = pd.DataFrame()  # object to collect parameter series
    for maturity in maturity_list:
        first = True
        for pricing_date in pricing_date_list:
            # select relevant option quotes
            option_selection, forward = srd_get_option_selection(pricing_date,
                                                             maturity)
            # instantiate all model given option selection
            vstoxx_model, option_models = srd_get_option_models(pricing_date,
                                                            maturity,
                                                            option_selection)
            if first:
                # global optimization to start with
                opt = spo.brute(srd_mean_squared_error,
                    ((1.25, 6.51, 0.75),   # range for kappa
                     (10., 20.1, 2.5),   # range for theta
                     (0.5, 10.51, 2.5)),  # range for volatility
                     finish=None)
            # local optimization
            opt = spo.fmin(srd_mean_squared_error, opt,
                           maxiter=550, maxfun=650,
                           xtol=0.0000001, ftol=0.0000001);
            # calculate MSE for storage
            MSE = srd_mean_squared_error(opt)
            # store main parameters and results
            parameters = parameters.append(
                     pd.DataFrame(
                     {'date' : pricing_date,
                      'maturity' : maturity,
                      'initial_value' : vstoxx_model.initial_value,
                      'kappa' : opt[0],
                      'theta' : opt[1],
                      'sigma' : opt[2],
                      'MSE' : MSE},
                      index=[0]), ignore_index=True)
            first = False  # set to False after first iteration
            last = opt  # store optimal parameters for reference
            print ("Maturity %s" % str(maturity)[:10]
                   + " | Pricing Date %s" % str(pricing_date)[:10]
                   + " | MSE %6.5f" % MSE)
    return parameters

if __name__ == '__main__':
    t0 = time.time()
    # define the dates for the calibration
    pricing_date_list = pd.date_range('2014/1/2', '2014/3/31', freq='B')
    # select the maturities
    maturity_list = [third_fridays[3]]  # only 18. April 2014 maturity
    # start the calibration
    parameters = srd_get_parameter_series(pricing_date_list, maturity_list)
    # plot the results
    for mat in maturity_list:
        # fig1, ax1 = plt.subplots()
        to_plot = parameters[parameters.maturity ==
                         maturity_list[0]].set_index('date')[
                        ['kappa', 'theta', 'sigma', 'MSE']]
        to_plot.plot(subplots=True, color='b', figsize=(10, 12),
                 title='SRD | ' + str(mat)[:10])
        plt.tight_layout()
        plt.savefig('./images/dx_srd_cali_1_%s_.png' % str(mat)[:10])
        # plotting the histogram of the MSE values
        fig, ax = plt.subplots()
        dat = parameters.MSE
        dat.hist(bins=30, ax=ax)
        plt.axvline(dat.mean(), color='r', ls='dashed',
                        lw=1.5, label='mean = %5.4f' % dat.mean())
        plt.legend()
        plt.tight_layout()
        plt.savefig('./images/dx_srd_cali_1_hist_%s_.png' % str(mat)[:10])
    # measuring and printing the time needed for the script execution
    print('Time in minutes %.2f' % ((time.time() - t0) / 60))
