release: python manage.py migrate
web: waitress-serve --port=$PORT settings.wsgi:application
rebalancer: python manage.py rebalancer
doge_trainer: python manage.py train_doge
doge_trader: python manage.py run_doge_strategies