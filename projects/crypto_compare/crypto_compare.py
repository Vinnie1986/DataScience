import json
from datetime import datetime
from datetime import timedelta

import requests
from dateutil import parser

url = "http://api.coindesk.com/v1/bpi/historical/close.json"

querystring = {"start": "2013-09-01", "end": "2017-07-05"}

headers = {
    'cache-control': "no-cache",
}

response = requests.request("GET", url, headers=headers, params=querystring)

currency_price = json.loads(response.text).get('bpi')

url = "https://api.github.com/repos/bitcoin/bitcoin/stats/commit_activity"

headers = {
    'cache-control': "no-cache",
}

response = requests.request("GET", url, headers=headers)

currency_commits = json.loads(response.text)


def convert_unix_ts(currency_commits):
    for element in currency_commits:
        unix_ts = element.get('week')
        element['week'] = datetime.fromtimestamp(unix_ts)
    return currency_commits


def convert_commits_to_daily_values(currency_commits):
    ret_value = {}
    for element in currency_commits:
        for i, day_value in enumerate(element.get('days')):
            date_ = (element.get('week') + timedelta(days=i)).date().strftime('%Y-%m-%d')
            ret_value[date_] = day_value

    return ret_value


currency_commits = convert_unix_ts(currency_commits)
currency_commits = convert_commits_to_daily_values(currency_commits)

import matplotlib.pyplot as plt
import math
import numpy as np


def create_plots(time_series, cum_sum=False):
    date_list = [parser.parse(element) for element in list(time_series.keys())]
    x = []
    for element in date_list:
        if element >= datetime(year=2016, month=7, day=24):
            x.append(element)

    y = []
    if cum_sum:
        values_list = np.cumsum(list(time_series.values()))
    else:
        values_list = list(time_series.values())
    for element in values_list:
        try:
            if element > 0:
                y.append(math.log(element))
            else:
                y.append(element)
        except Exception:
            pass
    y = y[-len(x):]
    plt.plot(x, y)


create_plots(currency_commits)
create_plots(currency_price)
plt.show()
