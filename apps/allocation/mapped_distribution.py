from apps.TA.storages.abstract.ticker import TickerStorage as MassiveShit


from collections import namedtuple

VoteFraction = namedtuple("VoteFraction", "transaction_currency_vote_fraction, counter_currency_vote_fraction")


def votes_on(ticker="BTC_USDT"):
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



    def get_allocations(self):
        allocations = {}

        votes_on_tickers = {ticker: votes_on(ticker) for ticker in self.tickers}

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

        print(allocations)
        from functools import reduce
        alloc_sum = sum([reduce(lambda x, y: x*y, value) for key, value in allocations.items()])


        for key, value in allocations.items():
            alloc = (reduce(lambda x, y: x*y, value)/float(alloc_sum))*100
            print(f"{key}: {reduce(lambda x, y: x*y, value):.2f}      Norm: {alloc:.2f}%\n")


        return allocations





from apps.allocation.mapped_distribution import MappedDistribution as MP
mp  = MP()
mp.get_allocations()
