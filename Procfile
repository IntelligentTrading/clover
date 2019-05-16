release: python manage.py migrate
web: waitress-serve --port=$PORT settings.wsgi:application

data_worker: python manage.py fetch_ohlc_tickers.py
TA_worker: python manage.py TA_worker
TA_data_restore: python manage.py TA_restore
TA_data_fill_gaps: python manage.py TA_fill_gaps

# doge_trainer: python manage.py train_doge
doge_trader: python manage.py run_doge_strategies
rebalancer: python manage.py rebalancer
