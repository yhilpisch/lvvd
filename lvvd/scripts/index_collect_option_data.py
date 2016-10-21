#
# Module to collect option series data
# from the web
# Source: www.eurexchange.com
# Data is needed to calculate the VSTOXX
# and its sub-indexes
#
# (c) Dr. Yves J. Hilpisch
# Listed Volatility and Variance Derivatives
#
import requests
import datetime as dt
import pandas as pd
import numpy as np
from StringIO import *
from index_date_functions import *

#
# The URL template
#
url1 = 'http://www.eurexchange.com/action/exchange-en/'
url2 = '180106-180102/180102/onlineStats.do?productGroupId=846'
url3 = '&productId=19068&viewType=3&cp=%s&month=%s&year=%s&busDate=%s'
URL = url1 + url2 + url3


def collect_option_series(month, year, start):
    ''' Collects daily option data from web source.

    Parameters
    ==========
    month: int
        maturity month
    year: int
        maturity year
    start: datetime object
        starting date

    Returns
    =======
    dataset: pandas DataFrame object
        object containing the collected data
    '''
    end = dt.datetime.today()
    delta = (end - start).days

    dataset = pd.DataFrame()
    for t in range(0, delta):  # runs from start to today
        date = start + dt.timedelta(t)
        dummy = get_data(month, year, date)  # get data for one day
        if len(dummy) != 0:
            if len(dataset) == 0:
                dataset = dummy
            else:
                dataset = pd.concat((dataset, dummy))  # add data

    return dataset


def get_data(month, year, date):
    ''' Get the data for an option series.

    Parameters
    ==========
    month: int
        maturity month
    year: int
        maturity year
    date: datetime object
        the date for which the data is collected

    Returns
    =======
    dataset: pandas DataFrame object
        object containing call & put option data
    '''

    date_string = date.strftime('%Y%m%d')
    # loads the call data from the web
    data = get_data_from_www('Call', month, year, date_string)
    calls = parse_data(data, date)  # parse the raw data
    calls = calls.rename(columns={'Daily settlem. price': 'Call_Price'})

    calls = pd.DataFrame(calls.pop('Call_Price').astype(float))
    # the same for puts
    data = get_data_from_www('Put', month, year, date_string)
    puts = parse_data(data, date)
    puts = puts.rename(columns={'Daily settlem. price': 'Put_Price'})
    puts = pd.DataFrame(puts.pop('Put_Price').astype(float))

    dataset = merge_and_filter(puts, calls)   # merges the two time series

    return dataset


def get_data_from_www(oType, matMonth, matYear, date):
    ''' Retrieves the raw data of an option series from the web.

    Parameters
    ==========
    oType: string
        either 'Put' or 'Call'
    matMonth: int
        maturity month
    matYear: int
        maturity year
    date: string
        expiry in the format 'YYYYMMDD'

    Returns
    =======
    a: string
        raw text with option data
    '''

    url = URL % (oType, matMonth, matYear, date)  # parametrizes the URL
    a = requests.get(url).text
    return a


def merge_and_filter(puts, calls):
    ''' Gets two pandas time series for the puts and calls
    (from the same option series), merges them, filters out
    all options with prices smaller than 0.5 and
    returns the resulting DataFrame object.

    Parameters
    ==========
    puts: pandas DataFrame object
        put option data
    calls: pandas DataFrame object
        call option data

    Returns
    =======
    df: pandas DataFrame object
        merged & filtered options data
    '''

    df = calls.join(puts, how='inner')  # merges the two time series
    # filters all prices which are too small
    df = df[(df.Put_Price >= 0.5) & (df.Call_Price >= 0.5)]

    return df


def parse_data(data, date):
    ''' Parses the HTML table and transforms it into a CSV compatible
    format. The result can be directly imported into a pandas DataFrame.

    Parameters
    ==========
    data: string
        document containing the Web content
    date: datetime object
        date for which the data is parsed

    Returns
    =======
    dataset: pandas DataFrame object
        transformed option raw data
    '''
    parts = data.split('<table')
    parts2 = parts[1].split('</table')
    dummy = parts2[0].replace(' class="odd"','')
    dummy = dummy.replace(' class="even"','')
    parts3 = dummy.split('<tr><td><b>Total</b>')
    table = parts3[0]   # the html table containing the data
    table = table.replace('class="dataTable"><thead>', 'Pricing day')

    # replace tags by commas and newlines
    table = table.replace('</tr>', '\n')
    table = table.replace(',', '')
    table = table.replace('<td>', ',')
    table = table.replace('</td>', '')
    table = table.replace('<th>', ',')
    table = table.replace('</th>', '')
    table = table.replace('</thead><tbody>', '\n')

    # the resulting string looks like a CSV file
    date_string = date.strftime('%d.%m.%Y')
    table = table.replace('<tr>', date_string)

    string = StringIO(table)  # mask the string as file
    dataset = pd.read_csv(string, parse_dates=[0], index_col=(0, 1),
                dayfirst=True)  # read the 'file' as pandas DataFrame object

    return dataset


def data_collection(path):
    ''' Main function which saves data into the HDF5 file
    'index_option_series.h5' for later use.

    Parameters
    ==========
    path: string
        path to store the data
    '''
    # file to store data
    store = pd.HDFStore(path + 'index_option_series.h5', 'a')

    today = dt.datetime.today()
    start = today - dt.timedelta(31)  # the last 31 days

    day = start.day
    month = start.month
    year = start.year

    for i in range(4):  # iterates over the next 4 months
        dummy_month = month + i
        dummy_year = year
        if dummy_month > 12:
            dummy_month -= 12
            dummy_year += 1

        # collect daily data beginning 31 days ago (start) for
        # option series with expiry dummy_month, dummy_year
        dataset = collect_option_series(dummy_month, dummy_year, start)

        dummy_date = dt.datetime(dummy_year, dummy_month, day)

        # abbreviation for expiry date (for example Oct14)
        series_name = dummy_date.strftime('%b%y')

        if series_name in store.keys():  # if data for that series exists
            index_old = store[series_name].index
            index_new = dataset.index

            if len(index_new - index_old) > 0:
                dummy = pd.concat((store[series_name],
                     dataset.ix[index_new - index_old]))  # add the new data

                store[series_name] = dummy
        else:
            if len(dataset) > 0:
            # if series is new, write whole data set into data store
                store[series_name] = dataset

    store.close()
