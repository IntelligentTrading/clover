## Setting up autotrading
1. Configure the following variables in `settings/__init__.py`:
```
DOGE_RETRAINING_PERIOD_SECONDS = ONE_HOUR           # how often to retrain and reinit the committee
DOGE_TRAINING_PERIOD_DURATION_SECONDS = ONE_DAY     # the duration of the training period
DOGE_LOAD_ROCKSTARS = True                          # whether to load historically best performing traders
DOGE_MAX_ROCKSTARS = 20                             # how many rockstars to load
DOGE_REBALANCING_PERIOD_SECONDS = 20*60             # how often to rebalance the portfolio
```

2. Run the following two processes concurrently:
```
python manage.py TA_Worker              # subscribes to TA indicators and recomputes doge votes every 5 minutes
python manage.py doge_autotrade         # runs the rebalancer every DOGE_REBALANCING_PERIOD_SECONDS
                                        # also ensures that the committees are retrained every DOGE_RETRAINING_PERIOD_SECONDS
```
