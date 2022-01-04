#
# Valuation of European volatility options
# by Monte Carlo simulation in
# Gruenbichler and Longstaff (1996) model
# -- analysis of valuation results
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import time
import math
import numpy as np
from datetime import datetime
from srd_functions import generate_paths, call_price
from srd_simulation_results import *

# Model Parameters
v0 = 20.0  # initial volatility
kappa = 3.0  # speed of mean reversion
theta = 20.0  # long-term volatility
sigma = 3.2  # standard deviation coefficient
zeta = 0.0  # factor of the expected volatility risk premium
r = 0.01  # risk-free short rate

# General Simulation Parameters
write = True
var_red = [(False, False), (False, True), (True, False), (True, True)]
    # 1st = mo_match -- random number correction (std + mean + drift)
    # 2nd = anti_paths -- antithetic paths for variance reduction
# number of time steps
steps_list = [25, 50, 75, 100]
# number of paths per valuation
paths_list = [2500, 50000, 75000, 100000, 125000, 150000]
SEED = 100000  # seed value
runs = 3  # number of simulation runs
PY1 = 0.010  # performance yardstick 1: abs. error in currency units
PY2 = 0.010  # performance yardstick 2: rel. error in decimals
maturity_list = [1.0 / 12 , 1.0 / 4, 1.0 / 2, 1.0]  # maturity list
strike_list = [15.0, 17.5, 20.0, 22.5, 25.0]  # strike list


def generate_paths(x0, kappa, theta, sigma, T, M, I):
    ''' Simulation of square-root diffusion with exact discretization

    Parameters
    ==========
    x0: float (positive)
        starting value
    kappa: float (positive)
        mean-reversion factor
    theta: float (positive)
        long-run mean
    sigma: float (positive)
        volatility (of volatility)
    T: float (positive)
        time-to-maturity
    M: int
        number of time intervals
    I: int
        number of simulation paths

    Returns
    =======
    x: NumPy ndarray object
        simulated paths
    '''
    dt = float(T) / M
    x = np.zeros((M + 1, I), dtype=np.float)
    x[0, :] = x0
    # matrix filled with standard normally distributed rv
    ran = randoms(M, I)
    d = 4 * kappa * theta / sigma ** 2
    # constant factor in the integrated process of x
    c = (sigma ** 2 * (1 - math.exp(-kappa * dt))) / (4 * kappa)
    if d > 1:
        for t in range(1, M + 1):
            # non-centrality parameter
            l = x[t - 1, :] * math.exp(-kappa * dt) / c
            # matrix with chi-squared distributed rv
            chi = np.random.chisquare(d - 1, I)
            x[t, :] = c * ((ran[t] + np.sqrt(l)) ** 2 + chi)
    else:
        for t in range(1, M + 1):
            l = x[t - 1, :] * math.exp(-kappa * dt) / c
            N = np.random.poisson(l / 2, I)
            chi = np.random.chisquare(d + 2 * N, I)
            x[t, :] = c * chi
    return x


def randoms(M, I):
    ''' Function to generate pseudo-random numbers with variance reduction.

    Parameters
    ==========
    M: int
        number of discrete time intervals
    I: int
        number of simulated paths

    Returns
    =======
    rand: Numpy ndarray object
        object with pseudo-random numbers
    '''
    if anti_paths is True:
        rand_ = np.random.standard_normal((M + 1, I / 2))
        rand = np.concatenate((rand_, -rand_), 1)
    else:
        rand = np.random.standard_normal((M + 1, I))
    if mo_match is True:
        rand = rand / np.std(rand)
        rand = rand - np.mean(rand)
    return rand


t0 = time.time()
sim_results = pd.DataFrame()

for vr in var_red:  # variance reduction techniques
    mo_match, anti_paths = vr
    for M in steps_list:  # number of time steps
        for I in paths_list:  # number of paths
            t1 = time.time()
            d1 = datetime.now()
            abs_errors = []
            rel_errors = []
            l = 0.0
            errors = 0
            # name of the simulation setup
            name = ('Call_' + str(runs) + '_'
                    + str(M) + '_' + str(I / 1000)
                    + '_' + str(mo_match)[0] + str(anti_paths)[0] +
                    '_' + str(PY1 * 100) + '_' + str(PY2 * 100))
            np.random.seed(SEED)  # RNG seed value
            for run in range(runs):  # simulation runs
                print "\nSimulation Run %d of %d" % (run + 1, runs)
                print "----------------------------------------------------"
                print ("Elapsed Time in Minutes %8.2f"
                        % ((time.time() - t0) / 60))
                print "----------------------------------------------------"
                z = 0
                for T in maturity_list:  # time-to-maturity
                    dt = T / M  # time interval in year fractions
                    V = generate_paths(v0, kappa, theta, sigma, T, M, I)
                        # volatility process paths
                    print "\n  Results for Time-to-Maturity %6.3f" % T
                    print "  -----------------------------------------"
                    for K in strike_list:  # Strikes
                        h = np.maximum(V[-1] - K, 0)  # inner value matrix
                        # MCS estimator
                        call_estimate = math.exp(-r * T) * np.sum(h) / I * 100
                        # BSM analytical value
                        callalue = call_price(v0, kappa, theta, sigma,
                                        zeta, T, r, K) * 100
                        # errors
                        diff = call_estimate - callalue
                        rdiff = diff / callalue
                        abs_errors.append(diff)
                        rel_errors.append(rdiff * 100)
                        # output
                        br = "    ----------------------------------"
                        print "\n  Results for Strike %4.2f\n" % K
                        print ("    European Op. Value MCS    %8.4f" %
                                    call_estimate)
                        print ("    European Op. Value Closed %8.4f" %
                                    callalue)
                        print "    Valuation Error (abs)     %8.4f" % diff
                        print "    Valuation Error (rel)     %8.4f" % rdiff
                        if abs(diff) < PY1 or abs(diff) / callalue < PY2:
                                print "      Accuracy ok!\n" + br
                                CORR = True
                        else:
                                print "      Accuracy NOT ok!\n" + br
                                CORR = False
                                errors = errors + 1
                        print "    %d Errors, %d Values, %.1f Min." \
                                % (errors, len(abs_errors),
                            float((time.time() - t1) / 60))
                        print ("    %d Time Intervals, %d Paths"
                                % (M, I))
                        z = z + 1
                        l = l + 1

            t2 = time.time()
            d2 = datetime.now()
            if write is True:  # append simulation results
                sim_results = write_results(sim_results, name, SEED,
                        runs, M, I, mo_match, anti_paths,
                        l, PY1, PY2, errors,
                        float(errors) / l, np.array(abs_errors),
                        np.array(rel_errors), t2 - t1, (t2 - t1) / 60, d1, d2)

if write is True:
    # write/append DataFrame to HDFStore object
    write_to_database(sim_results)
