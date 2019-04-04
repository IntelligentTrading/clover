import logging
from copy import deepcopy
from datetime import datetime, timedelta
from apps.TA import HORIZONS, PERIODS_1HR
from apps.doge.doge_TA_actors import CommitteeVoteStorage
from settings import SUPPORTED_DOGE_TICKERS
from apps.backtesting.utils import datetime_from_timestamp

(SHORT_HORIZON, MEDIUM_HORIZON, LONG_HORIZON) = list(range(3))
(POLONIEX, BITTREX, BINANCE, BITFINEX, KUCOIN, GDAX, HITBTC) = list(range(7))

class NoCommitteeVotesFoundException(Exception):
    pass


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
    """
    Queries CommitteeVoteStorage for all supported tickers, collects committee votes and then distributes the
    allocations accordingly.
    :param at_datetime: datetime for which to query the CommitteeVoteStorage (if set to None, the latest available
    vote in the storage will be used)
    :return: a dictionary of allocations
    """
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

    committees_used = {}

    for ticker in tickers:
        for horizon in horizons:

            committee_ids = []
            # find the latest vote
            query_result = CommitteeVoteStorage.query(
                ticker=ticker, exchange=exchange, timestamp=now_datetime.timestamp(),
                periods_range=PERIODS_1HR*horizon_periods[horizon],
                periods_key=PERIODS_1HR*horizon_periods[horizon]
            )

            if len(query_result['scores']) == 0:   # no recent committees found
                raise NoCommitteeVotesFoundException(f'No recent committee votes found for ticker '
                                                     f'{ticker}, horizon {horizon} and time {now_datetime} '
                                                     f'(are you running TA_worker?)')


            for score, weighted_vote in zip(query_result['scores'], query_result['values']):
                timestamp = CommitteeVoteStorage.datetime_from_score(score)
                logging.info(f'           Loaded committee votes for {ticker} with horizon {horizon} at {timestamp}')

                time_weight = float(1) - (
                        (now_datetime - timestamp).total_seconds() / (
                        timedelta(hours=1) * horizon_periods[horizon] *
                        horizon_life_spans[horizon]).total_seconds())

                split_weighted_vote = str(weighted_vote).split(':')
                if len(split_weighted_vote) == 2:   # we have committee ids too
                    weighted_vote = float(split_weighted_vote[0])
                    committee_id = split_weighted_vote[1]
                    committee_ids.append(committee_id)

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
            committees_used[ticker] = {
                horizon: committee_ids
            }


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
    logging.debug(f'Final SUM of allocations for doge: {round(sum([a["portion"] for a in allocations_list])*100,3)}%')



    return allocations_list, committees_used

