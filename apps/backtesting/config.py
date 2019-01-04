'''
Legacy MySql connection
mysql_database_config = {
  'user': 'root',
  'password': 'itfscience',
  'host': '127.0.0.1',
  'database': 'prodclone_core_2018_06_10',
  'raise_on_warnings': True,
}
'''
import redis

backtesting_cols_to_names = {
    "strategy": "Strategy",
    "source": "Exchange",
    "utilized_signals": "Signals",
    "transaction_currency": "Transaction currency",
    "counter_currency": "Counter currency",
    "profit_percent": "Profit percent (BTC)",
    "profit_percent_USDT": "Profit percent (USDT)",
    "buy_and_hold_profit_percent": "Buy and hold profit percent (BTC)",
    "buy_and_hold_profit_percent_USDT": "Buy and hold profit percent (USDT)",
    "num_trades": "Number of trades",
    "num_buys": "Number of buys",
    "num_sells": "Nuber of sells",
    "mean_buy_sell_pair_return": "Mean buy-sell pair return",
    "sharpe_ratio": "Sharpe ratio",
    "start_time": "Start time",
    "end_time": "End time",
    "resample_period": "Resample period"
}

backtesting_report_columns = list(backtesting_cols_to_names.keys())
backtesting_report_column_names = list(backtesting_cols_to_names.values())


transaction_cost_percents = {
    "binance": 0.001
}

# top 20 altcoins on Coinmarketcap (USDT not included)
COINMARKETCAP_TOP_20_ALTS = "ETH,XRP,BCH,EOS,XLM,LTC,ADA,MIOTA,XMR,TRX,DASH,ETC,NEO,BNB,VET,XEM,XTZ,ZEC,OMG,LSK".split(",")

INF_CASH = 100000000000
INF_CRYPTO = 100000000000

ENABLE_BACKTEST_CACHE = False
CACHE_MODE_REDIS, CACHE_MODE_DICTIONARY = ("redis", "dictionary")
CACHE_MODE = CACHE_MODE_DICTIONARY
if ENABLE_BACKTEST_CACHE and CACHE_MODE == CACHE_MODE_REDIS:
    redis_instance = redis.Redis(host='localhost', port=6379, db=0)
else:
    redis_instance = None

POOL_SIZE = 12
