import os
import logging
import redis
from apps.TA import deployment_type
from settings import DEBUG, BACKTESTING_MODE

SIMULATED_ENV = deployment_type == "LOCAL"
# todo: use this to mark keys in redis db, so they can be separated and deleted

logger = logging.getLogger('redis_db')


if deployment_type == "LOCAL":
    from settings.local_settings import TA_REDIS_URL
    if TA_REDIS_URL and not BACKTESTING_MODE:
        database = redis.from_url(TA_REDIS_URL)
    else:
        REDIS_HOST, REDIS_PORT = "127.0.0.1:6379".split(":")
        pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0)
        database = redis.Redis(connection_pool=pool)
else:
    database = redis.from_url(os.environ.get("TA_REDIS_URL"))

def get_used_memory_percent():
    """

    :return: a number between 0 and 1 representing the percent of memory taken
    """
    used_memory, maxmemory = int(database.info()['used_memory']), int(database.info()['maxmemory'])
    return used_memory/maxmemory if maxmemory != 0 else 0




if DEBUG:
    logger.info("Redis connection established for app database.")
    maxmemory_human = database.info()['maxmemory_human']
    logger.info(f"Redis currently consumes {round(100*get_used_memory_percent(), 2)}% out of {maxmemory_human}")

