import datetime
import logging

from apps.TA.storages.data.price import PriceStorage
from apps.backtesting.data_sources import STORAGE_CLASS

from apps.backtesting.utils import datetime_from_timestamp
from apps.doge.doge_TA_actors import CommitteeStorage
from apps.doge.doge_utils import view_keys
from apps.portfolio.services.doge_votes import get_allocations_from_doge
from settings import logger
from settings.redis_db import database


def list_all_committees(ticker='BTC_USDT', exchange='binance'):
    values = database.zrange(f'{ticker}:{exchange}:CommitteeStorage', 0, -1)
    logger.info('Existing committees:')
    for item in values:
        item = item.decode('UTF8').split(':')
        timestamp = CommitteeStorage.timestamp_from_score(item[-1])
        logger.info(f'  ->  at timestamp {datetime_from_timestamp(timestamp)}')

        logger.info(f'This committee produced the following allocations: '
                    f'{get_allocations_from_doge(at_datetime=datetime.datetime.utcfromtimestamp(timestamp))}')


def get_indicator_status(indicator_key='Willr', ticker='BTC_USDT', exchange='binance'):
    indicator_keys = view_keys(f'{ticker}:{exchange}:{indicator_key}*')
    for key in indicator_keys:
        last_entry = database.zrange(key, -1, -1)[0]
        timestamp = CommitteeStorage.timestamp_from_score(last_entry.decode('UTF8').split(':')[-1])
        logger.info(f'For key {key}, last entry is at {datetime_from_timestamp(timestamp)}')


class RedisTests:

    @staticmethod
    def test_ticker_storages(ticker):
        from settings.redis_db import database
        from apps.backtesting.utils import datetime_from_timestamp

        params = dict(
            ticker=ticker,
            exchange="binance",
        )

        for indicator_type in STORAGE_CLASS:
            sorted_set_key = STORAGE_CLASS[indicator_type].compile_db_key(key=None,
                                                                          key_prefix=f"{params['ticker']}"
                                                                          f":{params['exchange']}:",
                                                                          key_suffix='*')
            logging.info(f'Processing indicator {sorted_set_key}')
            if len(database.keys(sorted_set_key)) == 0:
                logging.info('   no data.')
            for key in database.keys(sorted_set_key):
                query_response = database.zrange(key, 0, -1)
                if len(query_response) == 0:
                    logging.info(f'   {key}: no data')
                    continue
                try:
                    score_start = query_response[0].decode("utf-8").split(":")[-1]
                    score_end = query_response[-1].decode("utf-8").split(":")[-1]
                    time_start = datetime_from_timestamp(
                        STORAGE_CLASS[indicator_type].timestamp_from_score(score_start))
                    time_end = datetime_from_timestamp(STORAGE_CLASS[indicator_type].timestamp_from_score(score_end))
                    num_values = len(query_response)
                    logging.info(f'   {key}: {num_values} values, start time = {time_start}, end time = {time_end}')
                except Exception as e:
                    logging.info(f'Error decoding input: {str(e)}')
        return


    @staticmethod
    def find_gaps(key_pattern, start_timestamp, end_timestamp):
        from settings.redis_db import database

        start_score = PriceStorage.score_from_timestamp(start_timestamp)
        end_score = PriceStorage.score_from_timestamp(end_timestamp)

        keys = database.keys(key_pattern)
        for key in keys:
            logging.info(f'Processing data for {key}...')
            values = database.zrangebyscore(key, min=start_score, max=end_score)
            gaps = []
            for i, item in enumerate(values):
                if i == len(values) - 1:  # the last element
                    break
                current_score = int(item.decode('UTF8').split(':')[-1])
                next_score = int(values[i + 1].decode('UTF8').split(':')[-1])
                if next_score == current_score:
                    logging.warning(f'     Encountered duplicate scores: {item} and {values[i + 1]}')
                    continue
                if next_score != current_score + 1:
                    gaps.append((current_score, next_score))

            logging.info('Found gaps: ')
            from apps.backtesting.utils import datetime_from_timestamp
            for gap in gaps:
                start = datetime_from_timestamp(PriceStorage.timestamp_from_score(gap[0]))
                end = datetime_from_timestamp(PriceStorage.timestamp_from_score(gap[1]))
                logging.info(f'    start: {start}, end: {end}  (scores {gap[0]}-{gap[1]})')
