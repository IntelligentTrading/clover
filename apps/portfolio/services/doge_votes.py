import logging
from copy import deepcopy

import requests
from datetime import datetime, timedelta
from django.core.cache import cache

from apps.common.utilities.multithreading import start_new_thread
from settings import ITF_CORE_API_URL, ITF_CORE_API_KEY

(SHORT_HORIZON, MEDIUM_HORIZON, LONG_HORIZON) = list(range(3))
(POLONIEX, BITTREX, BINANCE, BITFINEX, KUCOIN, GDAX, HITBTC) = list(range(7))
from apps.TA import HORIZONS, PERIODS_1HR
from apps.doge.doge_TA_actors import DogeStorage


def get_allocations_from_doge(at_datetime=None):
    now_datetime = at_datetime or datetime.now()

    horizon_periods = {
        SHORT_HORIZON: 1,
        MEDIUM_HORIZON: 4,
        LONG_HORIZON: 24,
    }
    horizon_life_spans = {
        SHORT_HORIZON: 4,
        MEDIUM_HORIZON: 4,
        LONG_HORIZON: 4,
    }
    horizon_weights = {
        SHORT_HORIZON: 1,
        MEDIUM_HORIZON: 1,
        LONG_HORIZON: 1,
    }


    # fix horizon for now
    horizons = [SHORT_HORIZON]
    source = BINANCE
    tickers = ['BTC_USDT']
    exchange = 'binance'

    BTC_minimum_reserve = 0.0090
    BNB_minimum_reserve = 0.0010
    USDT_minimum_reserve = 0.0

    tickers_dict = {
        "BTC_USDT": {
            "coin": "BTC",
            "vote": 0,
            "portion": BTC_minimum_reserve,  # hold by default
        },
        "USDT_BTC": {
            "coin": "USDT",
            "vote": 0,
            "portion": USDT_minimum_reserve,  # hold by default
        },
        "BNB_BTC": {
            "coin": "BNB",
            "vote": 0,
            "portion": BNB_minimum_reserve,  # hold by default
        }
    }


    # TODO remove mock data
    now_datetime = datetime.fromtimestamp(DogeStorage.timestamp_from_score(189215))

    for ticker in tickers:
        for horizon in horizons:
            # find the latest vote
            query_result = DogeStorage.query(
                ticker=ticker, exchange=exchange, timestamp=now_datetime.timestamp(),
                periods_range=PERIODS_1HR*horizon_periods[horizon]
            )

            # TODO: remove this mock data
            query_result = {
                'scores': [189213, 189214, 189215],
                'values': [0.27, 0.55, -0.34],
            }

            for score, weighted_vote in zip(query_result['scores'], query_result['values']):
                timestamp = DogeStorage.datetime_from_score(score)


                time_weight = float(1) - (
                        (now_datetime - timestamp).total_seconds() / (
                        timedelta(hours=1) * horizon_periods[horizon] *
                        horizon_life_spans[horizon]).total_seconds())

                # re-normalize weighted vote to interval [0, 1]
                weighted_vote = (1.0 + weighted_vote) / 2
                vote = float(weighted_vote) * horizon_weights[horizon] * time_weight

                if ticker in tickers_dict:
                    tickers_dict[ticker]["vote"] += vote
                else:
                    tickers_dict[ticker] = {"vote": vote}

                # counter-ticker sum
                counter_ticker = f'{ticker.split("_")[1]}_{ticker.split("_")[0]}'

                # note: vote and counter_vote sum to time_weight, not to 1, modify if needed
                counter_vote = (1 - weighted_vote) * time_weight
                if counter_ticker in tickers_dict:
                    tickers_dict[counter_ticker]["vote"] += counter_vote
                else:
                    tickers_dict[counter_ticker] = {"vote": counter_vote}


    # Remove tickers with negative votes
    for ticker, data in deepcopy(tickers_dict).items():
        if data["vote"] <= 0.01:
            del tickers_dict[ticker]

    votes_sum = sum([data["vote"] for ticker, data in tickers_dict.items()])
    logging.debug("First SUM of votes: " + str(votes_sum))

    # Remove tickers with low votes
    for ticker, data in deepcopy(tickers_dict).items():
        if data["vote"]/votes_sum <= 0.001:
            del tickers_dict[ticker]

    votes_sum = sum([data["vote"] for ticker, data in tickers_dict.items()])
    logging.debug("New SUM of votes: " + str(votes_sum))

    for ticker, data in deepcopy(tickers_dict).items():
        tickers_dict[ticker]["portion"] += (data["vote"] / votes_sum) // 0.0001 * 0.0001
        if tickers_dict[ticker]["portion"] < 0.0010:
            del tickers_dict[ticker]

    allocations_sum = sum([data["portion"] for ticker, data in tickers_dict.items()])
    logging.debug(f"preliminary SUM of allocations: {round(allocations_sum*100,3)}%")

    allocations_dict = {}
    for ticker, data in tickers_dict.items():
        if not data["coin"] in allocations_dict:
            allocations_dict[data["coin"]] = 0
        allocations_dict[data["coin"]] += data["portion"]

    allocations_dict["BNB"] = max([BNB_minimum_reserve, allocations_dict.get("BNB", 0)])
    allocations_dict["BTC"] = max([BTC_minimum_reserve, (0.9999 - BNB_minimum_reserve - allocations_sum + allocations_dict.get("BTC", 0))])

    allocations_list = [{"coin": coin, "portion": (portion // 0.0001 / 10000)} for coin, portion in allocations_dict.items()]
    logging.debug(f'Final SUM of allocations: {round(sum([a["portion"] for a in allocations_list])*100,3)}%')

    return allocations_list


def get_counter_currency_name(counter_currency_index):
    (BTC, ETH, USDT, XMR) = list(range(4))
    COUNTER_CURRENCY_CHOICES = (
        (BTC, 'BTC'),
        (ETH, 'ETH'),
        (USDT, 'USDT'),
        (XMR, 'XMR'),
    )
    return next((cc_name for index, cc_name in COUNTER_CURRENCY_CHOICES if index == counter_currency_index), None)
