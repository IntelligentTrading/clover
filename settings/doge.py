from settings import ONE_HOUR, ONE_DAY, ONE_WEEK

SUPPORTED_DOGE_TICKERS = ['BTC_USDT', 'ETH_USDT', 'ETH_BTC',]
ENABLE_TA_FOR_SUPPORTED_DOGE_TICKERS_ONLY = True

ENABLE_SHITCOIN_TRADING = False
COMMITTEE_VOTE_SCOPE = {
    'ETH_BTC': ['BNB_BTC', 'XRP_BTC', 'EOS_BTC', 'BCHABC_BTC', 'LTC_BTC', 'IOTA_BTC', 'NEO_BTC', 'BTT_BTC', 'IOST_BTC',
                'ADA_BTC', 'BCD_BTC', 'ATOM_BTC', 'TFUEL_BTC', 'MATIC_BTC', 'TRX_BTC', 'ONT_BTC', 'ZEC_BTC', 'XMR_BTC']
}

DOGE_RETRAINING_PERIOD_SECONDS = 1.5*ONE_HOUR           # how often to retrain and reinit the committee


DOGE_TRAINING_PERIOD_DURATION_SECONDS = ONE_DAY     # the duration of the training period
DOGE_LOAD_ROCKSTARS = True
DOGE_MAX_ROCKSTARS = 20
DOGE_REBALANCING_PERIOD_SECONDS = 20*60             # how often to run the rebalancer when autotrading
DOGE_ENFORCE_FRESH_VOTES = True                     # if set to True, fails if no recent committee votes exist
DOGE_COMMITTEES_EXPIRE = True                       # if set to True, enforces loading a new doge committee if the
                                                    # current committee is older than DOGE_RETRAINING_PERIOD_SECONDS
                                                    # NOTE: if this is False, it will trade using the most recent committee
                                                    # (even if it's e.g. a few months old!!!)

DOGE_FALLBACK_IF_UNABLE_TO_TRAIN = True             # falls back to simple buy/sell/ignore traders if unable
                                                    # to train a committee with minimum fitnesss
                                                    # (this typically happens when there's a bull run, so trading can't beat the benchmark)

DOGE_FALLBACK_BUY_SELL_THRESHOLD_PERCENT = 0.5      # if falling back to buy/sell/ignore, this will default to buy if
                                                    # benchmark returns in the training period are greater than this
                                                    # threshold, selling if the losses are greater than -threshold, and ignore otherwise

# list of tickers for which doges will vote
SUPPORTED_TICKERS = [
    # TODO: add the top 50 _BTC tickers, remove all the _ETH and _BNB and other keymah
]

def supported_shitcoins():
    shitcoins = []
    for key in COMMITTEE_VOTE_SCOPE:
        shitcoins += COMMITTEE_VOTE_SCOPE[key]
    return list(set(shitcoins))
