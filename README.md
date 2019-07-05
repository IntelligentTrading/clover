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


## Setting up Redis locally
1. Make sure that ITF Core is connected to production database on AWS
2. Make sure Redis is connected to your local Redis, not Aiven!!
3. If needed, set LOAD_TALIB to False in Core settings (it is not needed for data restoration)
4. Make sure your Redis server is running (type `redis-server` in the Terminal)
5. Run this:
```
from settings.redis_db import database as db
for key in db.keys():   # only if you want to 😬
    db.delete(key)    

```
6. Run `python manage.py TA_restore` in Core; this will restore all the ticker prices
7. Run `restore_indicators` from `TA_wipe_restore_indicators` in Clover (you will need to set the score range manually)
