import os
import logging

JAN_1_2017_TIMESTAMP = int(1483228800)
(PERIODS_1HR, PERIODS_4HR, PERIODS_24HR) = (12, 48, 288)  # num of 5 min samples
HORIZONS = [PERIODS_1HR, ] # only run Clover on 1hr periods for now. #todo: add later

PRICE_INDEXES = ['open_price', 'close_price', 'low_price', 'high_price', 'midpoint_price', 'mean_price', 'price_variance',]
VOLUME_INDEXES = ['open_volume', 'close_volume', 'low_volume', 'high_volume',]

deployment_type = os.environ.get('DEPLOYMENT_TYPE', 'LOCAL')
if deployment_type == 'LOCAL':
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('core.apps.TA')


class TAException(Exception):
    def __init__(self, message):
        self.message = message
        logger.error(message)

class SuchWowException(Exception):
    def __init__(self, message):
        self.message = message
        such_wow = "==============SUCH=====WOW==============="
        logger.error(f'\n\n{such_wow}\n\n{message}\n\n{such_wow}\n\n')
