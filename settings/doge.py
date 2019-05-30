ONE_WEEK = 60*60*24*7
ONE_DAY = 60*60*24
ONE_HOUR = 60*60

SUPPORTED_DOGE_TICKERS = ['BTC_USDT', 'ETH_USDT', ] # 'ETH_BTC',] # TODO add ETH_BTC when ready

DOGE_RETRAINING_PERIOD_SECONDS = ONE_HOUR           # how often to retrain and reinit the committee


DOGE_TRAINING_PERIOD_DURATION_SECONDS = ONE_DAY     # the duration of the training period
DOGE_LOAD_ROCKSTARS = True
DOGE_MAX_ROCKSTARS = 20
DOGE_REBALANCING_PERIOD_SECONDS = 20*60             # how often to run the rebalancer when autotrading
DOGE_ENFORCE_FRESH_VOTES = False                    # if set to True, fails if no recent committee votes exist
DOGE_COMMITTEES_EXPIRE = False                      # if set to True, enforces loading a new doge committee if the
                                                    # current committee is older than DOGE_RETRAINING_PERIOD_SECONDS
                                                    # NOTE: if this is False, it will trade using the most recent committee
                                                    # (even if it's e.g. a few months old!!!)

DOGE_FALLBACK_IF_UNABLE_TO_TRAIN = True             # falls back to simple buy/sell/ignore traders if unable
                                                    # to train a committee with minimum fitnesss
                                                    # (this typically happens when there's a bull run, so trading can't beat the benchmark)

DOGE_FALLBACK_BUY_SELL_THRESHOLD_PERCENT = 0.5      # if falling back to buy/sell/ignore, this will default to buy if
                                                    # benchmark returns in the training period are greater than this
                                                    # threshold, selling if the losses are greater than -threshold, and ignore otherwise