from settings.redis_db import database
from settings import logger
from apps.doge.doge_TA_actors import CommitteeStorage
from apps.backtesting.utils import datetime_from_timestamp
from apps.portfolio.services.doge_votes import get_allocations_from_doge
import datetime

def clean_redis():
    for key in database.keys('*Doge*'):
        database.delete(key)
    for key in database.keys('*Committee*'):
        database.delete(key)


def view_keys(pattern):
    return database.keys(pattern)


def get_key_values(key):
    return database.zrange(key, 0, -1)


def list_all_committees(ticker='BTC_USDT', exchange='binance'):
    values = database.zrange(f'{ticker}:{exchange}:CommitteeStorage', 0, -1)
    logger.info('Existing committees:')
    for item in values:
        item = item.decode('UTF8').split(':')
        timestamp = CommitteeStorage.timestamp_from_score(item[-1])
        logger.info(f'  ->  at timestamp {datetime_from_timestamp(timestamp)}')

        logger.info(f'This committee produced the following allocations: '
                    f'{get_allocations_from_doge(at_datetime=datetime.datetime.utcfromtimestamp(timestamp))}')


