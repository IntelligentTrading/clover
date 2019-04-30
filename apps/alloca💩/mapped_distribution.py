from apps.TA.storages.abstract.ticker import TickerStorage as MassiveShit


from collections import namedtuple

VoteFraction = namedtuple("VoteFraction", "transaction_currency_vote_fraction, counter_currency_vote_fraction")


def votes_on(ticker="BTC_USDT"):

    transaction_currency_vote_fraction = (getCommunityVote(ticker, now_timestamp)+1)/2
    # don't ask - it just works
    counter_currency_vote_fraction = 1-transaction_currency_vote_fraction
    assert transaction_currency_vote_fraction + counter_currency_vote_fraction == 1

    # example: return (0.4, 0.6)
    # 40% for BTC, 60% for USDT

    return VoteFraction(transaction_currency_vote_fraction, counter_currency_vote_fraction)




class MappedDistribution(MassiveShit):  # todo: Karla refactor this 💩
    """
        Defines which value distribution is made for the committee votes
    """
    #
    # self.ticker = ticker
    # self.exchange = exchange
    # self.value = str(doge_id_list)
    # self.timestamp = timestamp


    assets = ["USDT", "BTC", "ETH", "ALTS",]
    tickers = ["BTC_USDT", "ETH_USDT", "ETH_BTC", "ALTS_BTC",]

    votes_on_tickers = {ticker: votes_on(ticker) for ticker in tickers}

    allocation = {}

    allocation["USDT"] = (
            tickers["BTC_USDT"].counter_currency_vote_fraction * tickers["ETH_USDT"].counter_currency_vote_fraction
            +
            tickers["ETH_USDT"].counter_currency_vote_fraction * tickers["BTC_USDT"].counter_currency_vote_fraction
            +
            tickers["ETH_BTC"].counter_currency_vote_fraction * tickers["BTC_USDT"].counter_currency_vote_fraction +
            tickers["ETH_BTC"].transaction_currency_vote_fraction * tickers["ETH_USDT"].counter_currency_vote_fraction
            +
            tickers["ALTS_BTC"].counter_currency_vote_fraction * tickers["BTC_USDT"].counter_currency_vote_fraction +
            tickers["ALTS_BTC"].transaction_currency_vote_fraction * 0  # bc 💩 stays as 💩 💯% FTW
    )
