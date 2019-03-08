import logging
from copy import deepcopy
from datetime import datetime, timedelta
from apps.TA import HORIZONS, PERIODS_1HR
from apps.doge.doge_TA_actors import CommitteeVoteStorage
from settings import SUPPORTED_DOGE_TICKERS
from apps.backtesting.utils import datetime_from_timestamp

(SHORT_HORIZON, MEDIUM_HORIZON, LONG_HORIZON) = list(range(3))
(POLONIEX, BITTREX, BINANCE, BITFINEX, KUCOIN, GDAX, HITBTC) = list(range(7))


def fill_tickers_dict(supported_tickers, minimum_reserves):
    tickers_dict = {}
    for ticker in supported_tickers:
        transaction_currency, counter_currency = ticker.split('_')
        inverse_ticker = f'{counter_currency}_{transaction_currency}'
        tickers_dict[ticker] = {
            'coin': transaction_currency,
            'vote': 0,
            'portion': minimum_reserves[transaction_currency] if transaction_currency in minimum_reserves else 0
        }
        tickers_dict[inverse_ticker] = {
            'coin': counter_currency,
            'vote': 0,
            'portion': minimum_reserves[counter_currency] if counter_currency in minimum_reserves else 0
        }
    # add BNB_BTC
    if 'BNB' in minimum_reserves:
        tickers_dict['BNB_BTC'] = {
            'coin': 'BNB',
            'vote': 0,
            'portion': minimum_reserves['BNB']

        }
    return tickers_dict


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
    tickers = SUPPORTED_DOGE_TICKERS
    exchange = 'binance'

    minimum_reserves = {
        'BTC': 0.0090,
        'BNB': 0.0010,
        'USDT': 0
    }

    tickers_dict = fill_tickers_dict(SUPPORTED_DOGE_TICKERS, minimum_reserves)

    for ticker in tickers:
        for horizon in horizons:
            # find the latest vote
            query_result = CommitteeVoteStorage.query(
                ticker=ticker, exchange=exchange, timestamp=now_datetime.timestamp(),
                periods_range=PERIODS_1HR*horizon_periods[horizon],
                periods_key=PERIODS_1HR*horizon_periods[horizon]
            )

            for score, weighted_vote in zip(query_result['scores'], query_result['values']):
                timestamp = CommitteeVoteStorage.datetime_from_score(score)
                logging.info(f'           Loaded committee votes for {timestamp}')

                time_weight = float(1) - (
                        (now_datetime - timestamp).total_seconds() / (
                        timedelta(hours=1) * horizon_periods[horizon] *
                        horizon_life_spans[horizon]).total_seconds())

                # re-normalize weighted vote to interval [0, 1]
                weighted_vote = (1.0 + float(weighted_vote)) / 2
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

    allocations_dict["BNB"] = max([minimum_reserves['BNB'], allocations_dict.get("BNB", 0)])
    allocations_dict["BTC"] = max([minimum_reserves['BTC'],
                                   (0.9999 - minimum_reserves['BNB'] - allocations_sum + allocations_dict.get("BTC", 0))])

    allocations_list = [{"coin": coin, "portion": (portion // 0.0001 / 10000)} for coin, portion in allocations_dict.items()]
    logging.debug(f'Final SUM of allocations: {round(sum([a["portion"] for a in allocations_list])*100,3)}%')



    return allocations_list

