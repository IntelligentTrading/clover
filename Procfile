release: python manage.py migrate
web: waitress-serve --port=$PORT settings.wsgi:application
TA_worker: python manage.py TA_worker
TA_fill_gaps: python manage.py TA_fill_gaps
rebalancer: python manage.py rebalancer
doge_trainer: python manage.py train_doge
doge_trader: python manage.py run_doge_strategies