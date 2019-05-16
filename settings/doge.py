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