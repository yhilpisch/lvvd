#
# Valuation of European volatility options
# by Monte Carlo simulation in
# Gruenbichler and Longstaff (1996) model
# -- Creating a database for simulation results
# with pandas and PyTables
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# filname for HDFStore to save results
filename = "../data/simulation_results.h5"


def write_results(sim_results, name, SEED, runs, steps, paths, mo_match,
                  anti_paths, l, PY1, PY2, errors, error_ratio,
                  abs_errors, rel_errors, t1, t2, d1, d2):
    ''' Appends simulation results to pandas DataFrame df and returns it.

    Parameters
    ==========
    see srd_simulation_analysis.py

    Returns
    =======
    df: pandas DataFrame object
        updated results object
    '''
    results = {
    'sim_name': name,
    'seed': SEED,
    'runs': runs,
    'time_steps': steps,
    'paths': paths,
    'mo_match': mo_match,
    'anti_paths': anti_paths,
    'opt_prices': l,
    'abs_tol': PY1,
    'rel_tol': PY2,
    'errors': errors,
    'error_ratio': error_ratio,
    'aval_err': sum(abs_errors) / l,
    'abal_err': sum(abs(rel_errors)) / l,
    'time_sec': t1,
    'time_min': t2,
    'time_opt': t1 / l,
    'start_date': d1,
    'end_date': d2
    }
    df = pd.concat([sim_results,
                    pd.DataFrame([results])],
                    ignore_index=True)
    return df


def write_to_database(sim_results, filename=filename):
    ''' Write pandas DataFrame sim_results to HDFStore object.

    Parameters
    ==========
    sim_results: pandas DataFrame object
        object with simulation results
    filename: string
        name of the file for storage
    '''
    h5 = pd.HDFStore(filename, 'a')
    h5.append('sim_results', sim_results, min_itemsize={'values': 30},
               ignore_index=True)
    h5.close()


def print_results(filename=filename, idl=0, idh=50):
    ''' Prints valuation results in detailed form.

    Parameters
    ==========
    filename: string
        HDFStore with pandas.DataFrame with results
    idl: int
        start index value
    idh: int
        stop index value
    '''
    h5 = pd.HDFStore(filename, 'r')
    sim_results = h5['sim_results']
    br = "----------------------------------------------------"
    for i in range(idl, min(len(sim_results), idh + 1)):
        row = sim_results.iloc[i]
        print br
        print "Start Calculations  %32s" % row['start_date'] + "\n" + br
        print "ID Number           %32d" % i
        print "Name of Simulation  %32s" % row['sim_name']
        print "Seed Value for RNG  %32d" % row['seed']
        print "Number of Runs      %32d" % row['runs']
        print "Time Steps          %32d" % row['time_steps']
        print "Paths               %32d" % row['paths']
        print "Moment Matching     %32s" % row['mo_match']
        print "Antithetic Paths    %32s" % row['anti_paths'] + "\n"
        print "Option Prices       %32d" % row['opt_prices']
        print "Absolute Tolerance  %32.4f" % row['abs_tol']
        print "Relative Tolerance  %32.4f" % row['rel_tol']
        print "Errors              %32d" % row['errors']
        print "Error Ratio         %32.4f" % row['error_ratio'] + "\n"
        print "Aver Val Error      %32.4f" % row['aval_err']
        print "Aver Abs Val Error  %32.4f" % row['abal_err']
        print "Time in Seconds     %32.4f" % row['time_sec']
        print "Time in Minutes     %32.4f" % row['time_min']
        print "Time per Option     %32.4f" % row['time_opt'] + "\n" + br
        print "End Calculations    %32s" % row['end_date'] \
                 + "\n" + br + "\n"
    print "Total number of rows in table %d" % len(sim_results)
    h5.close()


def plot_error_ratio(filename=filename):
    ''' Show error ratio vs. paths * time_steps (i.e. granularity).

    Parameters
    ==========
    filename: string
        name of file with data to be plotted
    '''
    h5 = pd.HDFStore(filename, mode='r')
    sim_results = h5['sim_results']
    x = np.array(sim_results['paths'] * sim_results['time_steps'], dtype='d')
    x = x / max(x)
    y = sim_results['error_ratio']
    plt.plot(x, y, 'bo', label='error ratio')
    rg = np.polyfit(x, y, deg=1)
    plt.plot(np.sort(x), np.polyval(rg, np.sort(x)), 'r', label='regression',
             linewidth=2)
    plt.xlabel('time steps * paths (normalized)')
    plt.ylabel('errors / option valuations')
    plt.legend()
    plt.grid(True)
    h5.close()
