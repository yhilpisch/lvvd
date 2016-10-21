#
# Module with simulation functions for
# Square-Root Jump Diffusion (SRJD) model
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import math
import pickle
import numpy as np
import scipy.interpolate as scint

v0 = 17.6639  # initial VSTOXX index level

# parameters of square-root diffusion
kappa = 2.0  # speed of mean reversion
theta = 15.0  # long-term volatility
sigma = 1.0  # standard deviation coefficient

# parameters of log-normal jump
lamb = 0.4  # intensity (jumps per year)
mu = 0.4  # average jump size
delta = 0.1  # volatility of jump size

# general parameters
r = 0.01  # risk-free interest rate
K = 17.5  # strike
T = 0.5  # time horizon
M = 150  # time steps
I = 10000  # number of MCS paths
anti_paths = True  # antithetic variates
mo_match = True  # moment matching


# deterministic shift parameters
varphi = pickle.load(open('data/varphi'))
tck = scint.splrep(varphi['ttms'], varphi['varphi'], k=1)
  # linear splines interpolation of
  # term structure calibration differences


def random_number_gen(M, I, fixed_seed=False):
    ''' Generate standard normally distributed pseudo-random numbers

    Parameters
    ==========
    M: int
        number of time intervals
    I: int
        number of paths

    Returns
    =======
    ran: NumPy ndarrayo object
        random number array
    '''
    if fixed_seed is True:
        np.random.seed(10000)
    if anti_paths is True:
        ran = np.random.standard_normal((M + 1, I / 2))
        ran = np.concatenate((ran, -ran), axis=1)
    else:
        ran = np.standard_normal((M + 1, I))
    if mo_match is True:
        ran = ran / np.std(ran)
        ran -= np.mean(ran)
    return ran


def srjd_simulation(x0, kappa, theta, sigma,
                    lamb, mu, delta, T, M, I, fixed_seed=False):
    ''' Function to simulate square-root jump Difusion.

    Parameters
    ==========
    x0: float
        initial value
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    T: float
        time horizon/maturity
    M: int
        time steps
    I: int
        number of simulation paths

    Returns
    =======
    x: NumPy ndarray object
        array with simulated SRJD paths
    '''
    dt = float(T) / M  # time interval
    shift = scint.splev(np.arange(M + 1) * dt, tck, der=0)
      # deterministic shift values
    xh = np.zeros((M + 1, I), dtype=np.float)
    x = np.zeros((M + 1, I), dtype=np.float)
    xh[0, :] = x0
    x[0, :] = x0
    # drift contribution of jump p.a.
    rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1)
    # 1st matrix with standard normal rv
    ran1 = random_number_gen(M + 1, I, fixed_seed)
    # 2nd matrix with standard normal rv
    ran2 = random_number_gen(M + 1, I, fixed_seed)
    # matrix with Poisson distributed rv
    ran3 = np.random.poisson(lamb * dt, (M + 1, I))
    for t in range(1, M + 1):
        xh[t, :] = (xh[t - 1, :] +
                    kappa * (theta - np.maximum(0, xh[t - 1, :])) * dt
                  + np.sqrt(np.maximum(0, xh[t - 1, :])) * sigma
                  * ran1[t] * np.sqrt(dt)
                  + (np.exp(mu + delta * ran2[t]) - 1) * ran3[t]
                  * np.maximum(0, xh[t - 1, :]) - rj * dt)
        x[t, :] = np.maximum(0, xh[t, :]) + shift[t]
    return x


def srjd_call_valuation(v0, kappa, theta, sigma,
                        lamb, mu, delta, T, r, K, M=M, I=I,
                        fixed_seed=False):
    ''' Function to value European volatility call option in SRDJ model.
    Parameters see function srjd_simulation.

    Returns
    =======
    call_value: float
        estimator for European call present value for strike K
    '''
    v = srjd_simulation(v0, kappa, theta, sigma,
                        lamb, mu, delta, T, M, I, fixed_seed)
    call_value = np.exp(-r * T) * sum(np.maximum(v[-1] - K, 0)) / I
    return call_value

if __name__ is '__main__':
    call_value = srjd_call_valuation(v0, kappa, theta, sigma,
                                     lamb, mu, delta, T, r, K, M, I)
    print "Value of European call by MCS: %10.4f" % call_value
