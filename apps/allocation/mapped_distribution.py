from apps.TA.storages.abstract.ticker import TickerStorage as MassiveShit
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from apps.TA import PERIODS_1HR
from apps.doge.doge_TA_actors import CommitteeVoteStorage
from settings import SUPPORTED_DOGE_TICKERS
from collections import namedtuple
from apps.backtesting.utils import datetime_from_timestamp


(POLONIEX, BITTREX, BINANCE, BITFINEX, KUCOIN, GDAX, HITBTC) = list(range(7))

class NoCommitteeVotesFoundException(Exception):
    pass

CommitteeVote = namedtuple("CommitteeVote", "timestamp, vote, committee_id")


from collections import namedtuple

VoteFraction = namedtuple("VoteFraction", "transaction_currency_vote_fraction, counter_currency_vote_fraction")


def votes_on_test(ticker="BTC_USDT"):
    """

    :param ticker:
    :return: example: return VoteFraction(0.4, 0.6)  # 40% for BTC, 60% for USDT
    """

    # transaction_currency_vote_fraction = (getCommunityVote(ticker, now_timestamp)+1)/2
    # # don't ask - it just works
    # counter_currency_vote_fraction = 1-transaction_currency_vote_fraction
    # assert transaction_currency_vote_fraction + counter_currency_vote_fraction == 1

    # return VoteFraction(transaction_currency_vote_fraction, counter_currency_vote_fraction)

    results = {
                  "BTC_USDT": VoteFraction(0.4, 0.6),
                  "ETH_USDT": VoteFraction(0.3, 0.7),
                  "ETH_BTC": VoteFraction(0.75, 0.25),
                  "ALTS_BTC": VoteFraction(0.5, 0.5)
    }

    results = {
                  "BTC_USDT": VoteFraction(0.5, 0.5),
                  "ETH_USDT": VoteFraction(0.5, 0.5),
                  "ETH_BTC": VoteFraction(0.5, 0.5),
                  "ALTS_BTC": VoteFraction(0.5, 0.5)
    }

    results = {
                  "BTC_USDT": VoteFraction(0.2, 0.8),
                  "ETH_USDT": VoteFraction(0.5, 0.5),
                  "ETH_BTC": VoteFraction(0.8, 0.2),
                  "ALTS_BTC": VoteFraction(0.80, 0.2)
    }


    return results[ticker]



def votes_on(ticker, when_datetime=None):

    if ticker == 'ALTS_BTC' or ticker =='ETH_BTC':
        return VoteFraction(0.5, 0.5), []   # TODO we don't have data coming in for alts_btc or eth_btc yet, fix

    when_datetime = when_datetime or datetime.now()
    exchange = 'binance'
    query_result = CommitteeVoteStorage.query(
        ticker=ticker, exchange=exchange, timestamp=when_datetime.timestamp(),
        periods_range=PERIODS_1HR * 400,
        periods_key=PERIODS_1HR
    )

    if len(query_result['scores']) == 0:  # no recent committees found
        raise NoCommitteeVotesFoundException(f'No recent committee votes found for ticker '
                                             f'{ticker} and time {when_datetime} '
                                             f'(are you running TA_worker?)')

    committee_votes = []
    summary_vote = 0
    for score, weighted_vote in zip(query_result['scores'], query_result['values']):
        timestamp = CommitteeVoteStorage.datetime_from_score(score)
        logging.info(f'           Loaded committee votes for {ticker} at {timestamp}')

        time_weight = float(1) - (
                (when_datetime - timestamp).total_seconds() / (
            timedelta(hours=4)).total_seconds())


        time_weight = 1
        assert time_weight > 0


        committee_id = None
        split_weighted_vote = str(weighted_vote).split(':')
        if len(split_weighted_vote) == 2:  # we have committee ids too; this is legacy stuff because
            weighted_vote = float(split_weighted_vote[0])
            committee_id = split_weighted_vote[1]

        # re-normalize weighted vote to interval [0, 1]
        weighted_vote = (1.0 + float(weighted_vote)) / 2
        # print(weighted_vote)
        vote = float(weighted_vote) * time_weight
        committee_votes.append(CommitteeVote(committee_id=committee_id, timestamp=timestamp, vote=vote))
        summary_vote += vote

    summary_vote /= len(committee_votes)
    print(ticker, summary_vote)
    return VoteFraction(summary_vote, 1-summary_vote), committee_votes



class MappedDistribution(MassiveShit):  # todo: Karla refactor this 💩
    """
        Defines which value distribution is made for the committee votes
    """
    #
    # self.ticker = ticker
    # self.exchange = exchange
    # self.value = str(doge_id_list)
    # self.timestamp = timestamp

    def __init__(self):
        self.assets = ["USDT", "BTC", "ETH", "ALTS",]
        self.tickers = ["BTC_USDT", "ETH_USDT", "ETH_BTC", "ALTS_BTC",]

    def _reinforce_minimum_reserves(self, normalized_allocations):
        minimum_reserves = {
            'BTC': 0.03,  # 1%
            'BNB': 0.03  # .1%
        }
        reassigned_portion_amount = 0
        reassigned_tickers = []
        for ticker, portion in minimum_reserves.items():
            if ticker not in normalized_allocations or normalized_allocations[ticker] < portion:
                normalized_allocations[ticker] = portion
                reassigned_portion_amount += portion
                reassigned_tickers.append(ticker)

        total_non_reassigned = 0
        # now scale all non-reassigned tickers to accomodate new portions
        for ticker, allocation in normalized_allocations.items():
            if ticker not in reassigned_tickers:  # we need to scale
                total_non_reassigned += allocation

        for ticker in normalized_allocations:
            if ticker not in reassigned_tickers:
                normalized_allocations[ticker] = normalized_allocations[ticker] / total_non_reassigned * (
                            1 - reassigned_portion_amount)

        assert 0.99 < sum([allocation for _, allocation in normalized_allocations.items()]) <= 1

        return normalized_allocations

    def get_allocations(self):
        allocations = normalized_allocations = {}

        votes_on_tickers = {ticker: votes_on(ticker)[0] for ticker in self.tickers}

        allocations["USDT"] = [
                votes_on_tickers["BTC_USDT"].counter_currency_vote_fraction,
                votes_on_tickers["ETH_USDT"].counter_currency_vote_fraction,
                2
        ]

        # like what you see? here's more... 🤮🤮🤮

        allocations["BTC"] = [
                votes_on_tickers["BTC_USDT"].transaction_currency_vote_fraction,
                votes_on_tickers["ETH_BTC"].counter_currency_vote_fraction,
                votes_on_tickers["ALTS_BTC"].counter_currency_vote_fraction,
                4
        ]

        allocations["ETH"] = [
                votes_on_tickers["ETH_USDT"].transaction_currency_vote_fraction,
                votes_on_tickers["ETH_BTC"].transaction_currency_vote_fraction,
                2
        ]

        allocations["ALTS"] = [
                votes_on_tickers["ALTS_BTC"].transaction_currency_vote_fraction,
                len("Willr")/5
        ]


        from functools import reduce
        alloc_sum = sum([reduce(lambda x, y: x*y, value) for key, value in allocations.items()])


        for ticker, vote_fraction_list in allocations.items():
            alloc = (reduce(lambda x, y: x*y, vote_fraction_list)/float(alloc_sum))
            # print(f"{ticker}: {reduce(lambda x, y: x*y, vote_fraction_list):.2f}  Norm: {alloc:.2f}\n")
            normalized_allocations[ticker] = alloc

        # need to check minimum reserves for BTC and BNB



        # reformat normalized allocations as expected
        allocations_list = [{"coin": coin, "portion": (portion // 0.0001 / 10000)} for coin, portion in
                            normalized_allocations.items()]
        return normalized_allocations





mp  = MappedDistribution()
print(mp.get_allocations())
test_allocations = {
    'BTC': 0.03,
    'USDT': 0.94,
    'ETH': 0.04
}
print('With reinforced minimum reserves:')
print(mp._reinforce_minimum_reserves(test_allocations))
