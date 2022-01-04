#
# Module with functions for
# Gruenbichler and Longstaff (1996) model
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import math
import numpy as np
import scipy.stats as scs


def futures_price(v0, kappa, theta, zeta, T):
    ''' Futures pricing formula in GL96 model.

    Parameters
    ==========
    v0: float (positive)
        current volatility level
    kappa: float (positive)
        mean-reversion factor
    theta: float (positive)
        long-run mean of volatility
    zeta: float (positive)
        volatility risk premium
    T: float (positive)
        time-to-maturity

    Returns
    =======
    future: float
        price of a future
    '''
    alpha = kappa * theta
    beta = kappa + zeta
    future = (alpha / beta * (1 - math.exp(-beta * T))
                               + math.exp(-beta * T) * v0)
    return future


def cx(K, gamma, nu, lamb):
    ''' Complementary distribution function of non-central chi-squared density.

    Parameters
    ==========
    K: float (positive)
        strike price
    gamma: float (positive)
        as defined in the GL96 model
    nu: float (positive)
        degrees of freedom
    lamb: float (positive)
        non-centrality parameter

    Returns
    =======
    complementary distribution of nc cs density
    '''
    return 1 - scs.ncx2.cdf(gamma * K, nu, lamb)


def call_price(v0, kappa, theta, sigma, zeta, T, r, K):
    ''' Call option pricing formula in GL96 Model

    Parameters
    ==========
    v0: float (positive)
        current volatility level
    kappa: float (positive)
        mean-reversion factor
    theta: float (positive)
        long-run mean of volatility
    sigma: float (positive)
        volatility of volatility
    zeta: float (positive)
        volatility risk premium
    T: float (positive)
        time-to-maturity
    r: float (positive)
        risk-free short rate
    K: float(positive)
        strike price of the option

    Returns
    =======
    call: float
        present value of European call option
    '''
    D = math.exp(-r * T)  # discount factor

    alpha = kappa * theta
    beta = kappa + zeta
    gamma = 4 * beta / (sigma ** 2 * (1 - math.exp(-beta * T)))
    nu = 4 * alpha / sigma ** 2
    lamb = gamma * math.exp(-beta * T) * v0

    # the pricing formula
    call = (D * math.exp(-beta * T) * v0 * cx(K, gamma, nu + 4, lamb)
      + D * (alpha / beta) * (1 - math.exp(-beta * T))
      * cx(K, gamma, nu + 2, lamb)
      - D * K * cx(K, gamma, nu, lamb))
    return call


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
    # matrix filled with standard normal distributed rv
    ran = np.random.standard_normal((M + 1, I))
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


def call_estimator(v0, kappa, theta, sigma, T, r, K, M, I):
    ''' Estimation of European call option price in GL96 Model
    via Monte Carlo simulation

    Parameters
    ==========
    v0: float (positive)
        current volatility level
    kappa: float (positive)
        mean-reversion factor
    theta: float (positive)
        long-run mean of volatility
    sigma: float (positive)
        volatility of volatility
    T: float (positive)
        time-to-maturity
    r: float (positive)
        risk-free short rate
    K: float (positive)
        strike price of the option
    M: int
        number of time intervals
    I: int
        number of simulation paths

    Returns
    =======
    callvalue: float
        MCS estimator for European call option
    '''
    V = generate_paths(v0, kappa, theta, sigma, T, M, I)
    callvalue = math.exp(-r * T) * np.sum(np.maximum(V[-1] - K, 0)) / I
    return callvalue
