{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"http://hilpisch.com/tpq_logo.png\" alt=\"The Python Quants\" width=\"35%\" align=\"right\" border=\"0\"><br><br><br>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Listed Volatility and Variance Derivatives\n",
    "\n",
    "**Wiley Finance (2017)**\n",
    "\n",
    "Dr. Yves J. Hilpisch | The Python Quants GmbH\n",
    "\n",
    "http://tpq.io | [@dyjh](http://twitter.com/dyjh) | http://books.tpq.io\n",
    "\n",
    "<img src=\"https://hilpisch.com/images/lvvd_cover.png\" alt=\"Listed Volatility and Variance Derivatives\" width=\"30%\" align=\"left\" border=\"0\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DX Analytics &mdash; Square-Root Diffusion "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Introduction "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This chapter uses DX Analytics to model the VSTOXX volatility index by a square-root diffusion process as proposed in Gruenbichler and Longstaff (1996) and discussed in chapter _Valuing Volatility Derivatives_. It implements a study over a time period of three months to analyze how well the model performs in replicating market quotes for VSTOXX options."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Import and Selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data we are working with is for the first quarter of 2014. The complete data set is contained in the online resources accompanying this book. As usual, some imports first."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import datetime as dt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we read the data from the source into pandas ``DataFrame`` objects."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "h5 = pd.HDFStore('./data/vstoxx_march_2014.h5', 'r')\n",
    "vstoxx_index = h5['vstoxx_index']  # data for the index itself \n",
    "vstoxx_futures = h5['vstoxx_futures']  # data for the futures\n",
    "vstoxx_options = h5['vstoxx_options']  # data for the options\n",
    "h5.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Inspecting the data sub-set for the VSTOXX index itself, we see that we are dealing with 63 trading days."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "vstoxx_index.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "vstoxx_index.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Per trading day, there are eight futures quotes for the eight different maturities of the VSTOXX futures contract. This makes for a total of 504 futures quotes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "vstoxx_futures['DATE'] = pd.to_datetime(vstoxx_futures['DATE'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "vstoxx_futures['MATURITY'] = pd.to_datetime(vstoxx_futures['MATURITY'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "vstoxx_futures.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "vstoxx_futures.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By far the biggest data sub-set is the one for the VSTOXX options. There are for each trading day market quotes for puts and calls for eight different maturities and a multitude of different strike prices. This makes for a total of 46960 option quotes for the first quarter of 2014."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "vstoxx_options['DATE'] = pd.to_datetime(vstoxx_options['DATE'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "vstoxx_options['MATURITY'] = pd.to_datetime(vstoxx_options['MATURITY'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "vstoxx_options.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "vstoxx_options.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Maturity-wise we are dealing with a total of eleven dates. This is due to the fact that at any given time eight maturities for the VSTOXX futures and options contracts are available and we are looking at data for three months."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    }
   },
   "outputs": [],
   "source": [
    "third_fridays = sorted(set(vstoxx_futures['MATURITY']))\n",
    "third_fridays"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When it comes to the calibration of the square-root diffusion model for the VSTOXX, it is necessary to work with a selection from the large set of option quotes. The following function implements such a selection procedure, using different conditions to generate a sensible set of option quotes around the forward at-the-money level. The function `srd_get_option_selection()` is used in what follows to select the right sub-set of option quotes for each day during the calibration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('scripts')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import dx_srd_calibration as dxsrd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dxsrd.srd_get_option_selection??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Modeling the VSTOXX Options"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The previous chapter illustrates how European options are modeled with DX Analytics based on a geometric Brownian motion model (`dx.geometric_brownian_motion()`). To model the VSTOXX options for the calibration, we just need to replace that model by the square-root diffusion model `dx.square_root_diffusion()`. The respective market environment then needs some additional parameters."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "All the code used for the calibration is found in the Python script ``dx_srd_calibration.py`` (see the appendix for the complete script). After some imports, the script starts by defining some general parameters and curves for the market environment. During the calibration process, some of these get updated to reflect the current status of the optimization procedure. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!sed -n 11,53p scripts/dx_srd_calibration.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The function `srd_get_option_models()` creates valuation models for all options in a given selection of option quotes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dxsrd.srd_get_option_models??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calibration of the VSTOXX Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Calibration of a parametrized model usually boils down to using global and local optimization algorithms to find parameters that minimize a given target function. This process is discussed in detail in Hilpisch (2015, ch. 11). For the calibration process to follow, we use the helper function `srd_calculate_model_values()` to calculate \"at once\" the model values for the VSTOXX options at hand. The function parameter `p0` is a tuple since this is what the optimization functions provide as input."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dxsrd.srd_calculate_model_values??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The target function to be minimized during the calibration is the _mean-squared error_ of the model values given the market quotes of the VSTOXX options. Again, refer to Hilpisch (2015, ch. 11) for details and alternative formulations. The function `srd_mean_squared_error()` implements this concept and uses the function `srd_calculate_model_values()` for the option model value calculations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dxsrd.srd_mean_squared_error??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Equipped with the target function to be minimized, we can define the function for the global and local calibration routine itself. The calibration takes place for one or multiple maturities over the pricing date range defined. For example, the function `srd_get_parameter_series()` can calibrate the model (separately) for the two maturities May and June 2014 over the complete first quarter 2014."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dxsrd.srd_get_parameter_series??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The final step is to start the calibration and collect the calibration results. The calibration we implement is for the 18. April 2014 maturity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!sed -n 256,287p scripts/dx_srd_calibration.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### No Penalization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A visualization of the calibration results tells the whole story. The following figures shows the three square-root diffusion parameters over time and the resulting MSE values. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %run scripts/dx_srd_calibration.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"./images/dx_srd_cali_1_2014-04-18.png\" width=\"75%\">\n",
    "\n",
    "<p style=\"font-family: monospace;\">Square-root diffusion parameters and MSE values from the calibration to a single maturity (18. April 2014) &mdash; no penalization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see throughout, the results are quite good given the low MSE values. The mean MSE value is below 0.01 as seen in the following figure."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"images/dx_srd_cali_1_hist_2014-04-18.png\" width=\"75%\">\n",
    " \n",
    "<p style=\"font-family: monospace;\">Histogram of the mean-squared errors for the calibration of the square-root diffusion model to a single maturity (18. April 2015) &mdash; no penalization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### With Penalization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"./images/dx_srd_cali_1_2014-04-18_.png\" width=\"75%\">\n",
    "\n",
    "<p style=\"font-family: monospace;\">Square-root diffusion parameters and MSE values from the calibration to a single maturity (18. April 2014) &mdash; with penalization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see throughout, the results are quite good given the low MSE values. The mean MSE value is below 0.01 as seen in the following figure."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"images/dx_srd_cali_1_hist_2014-04-18_.png\" width=\"75%\">\n",
    " \n",
    "<p style=\"font-family: monospace;\">Histogram of the mean-squared errors for the calibration of the square-root diffusion model to a single maturity (18. April 2015) &mdash; with penalization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This chapter uses DX Analytics to model the VSTOXX volatility index by a square-root diffusion process. In similar vein, it is used to model traded European call options on the index to implement a calibration of the VSTOXX model over time. The results show that when calibrating the model to a single options maturity only, the model performs quite well yielding rather low MSE values throughout. The parameter values also seem to be in sensible regions throughout (e.g. `theta` between 15 and 18) and they evolve rather smoothly."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There exist closed form solutions for the price of a European call option in the square-root diffusion model of Gruenbichler and Longstaff (1996) as shown in chapter _Valuing Volatility Derivatives_. For our analysis in this chapter we have nevertheless used the Monte Carlo valuation model of DX Analytics since this approach is more general in that we can easily replace one model by another, maybe more sophisticated, one. This is done in the next chapter where the same study is implemented based on the square-root jump diffusion model presented in chapter _Advanced Modeling of the VSTOXX Index_. The only difference is that a few more parameters need to be taken care of."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Python Scripts"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### `dx_srd_calibration.py`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!cat scripts/dx_srd_calibration.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"http://hilpisch.com/tpq_logo.png\" alt=\"The Python Quants\" width=\"35%\" align=\"right\" border=\"0\"><br>\n",
    "\n",
    "<a href=\"http://tpq.io\" target=\"_blank\">http://tpq.io</a> | <a href=\"http://twitter.com/dyjh\" target=\"_blank\">@dyjh</a> | <a href=\"mailto:team@tpq.io\">team@tpq.io</a>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
