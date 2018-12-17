import logging
import time

from django.core.management.base import BaseCommand

from apps.TA.storages.utils.memory_cleaner import redisCleanup
from settings.redis_db import database

logger = logging.getLogger(__name__)

try:
    earliest_price_score = int(float(database.zrangebyscore("BTC_USDT:bittrex:PriceStorage:close_price", 0, "inf", 0, 1)[0].decode("utf-8").split(":")[0]))
except:
    from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
    earliest_price_score = TimeseriesStorage.score_from_timestamp(int(time.time()))


# todo for making this more efficient
# ✅ - only 5min price history, all else can be generated on demand
# ✅ def compress(timestamp): return (timestamp - JAN_1_2017_TIMESTAMP)/300
# 🚫 - floor all prices to 6 sig-figs (saving up to 6 digits for XX_USDT prices) on TickerStorage
# ✅  - but maybe no because we like operating with satoshis always
# ✅ - cast scores on indicators to integers (saving 2 digits)
# ✅ - use rabbitmq as a centralized task queue so workers can scale horizontally
# ✅ - reduce number of tickers being processed
# 
# firehose download historical data
# resampling and missing data
# Get pubsub thing working
# Send signals to SNS
# Confirm new signals are same as old
# Fully replace old signals with new
# turn signals into votes for portfolio
# Autotrade on portfolio
#
# Next week:
# Push all updates live
# Stop using old Aurora database

class Command(BaseCommand):
    help = 'Run Redis Subscribers for TA'

    def handle(self, *args, **options):
        logger.info("Starting TA worker.")

        subscribers = {}
        all_subscriber_classes = get_subscriber_classes() + get_doge_subscriber_classes()
        for subscriber_class in all_subscriber_classes:
            subscribers[subscriber_class.__name__] = subscriber_class()
            logger.debug(f'added subscriber {subscriber_class}')
            logger.debug(f'new subscriber is {subscribers[subscriber_class.__name__]}')

        for s in subscribers:
            logger.debug(f'latest channels: {subscribers[s].database.pubsub_channels()}')

        logger.info("Pubsub clients are ready.")

        while True:
            for class_name in subscribers:
                # logger.debug(f'checking subscription {class_name}: {subscribers[class_name]}')
                try:
                    subscribers[class_name]()  # run subscriber class

                except Exception as e:
                    logger.error(str(e))
                    logger.debug(subscribers[class_name].__dict__)

                time.sleep(0.001)  # be nice to the system :)


def get_subscriber_classes():

    from apps.TA.indicators.overlap import sma, ema, wma, dema, tema, trima, bbands, ht_trendline, kama, midprice
    from apps.TA.indicators.momentum import adx, adxr, apo, aroon, aroonosc, bop, cci, cmo, dx, macd, mom, ppo, \
        roc, rocr, rsi, stoch, stochf, stochrsi, trix, ultosc, willr

    return [

        # OVERLAP INDICATORS
        # midprice.MidpriceSubscriber,
        sma.SmaSubscriber, ema.EmaSubscriber, wma.WmaSubscriber,
        # dema.DemaSubscriber, tema.TemaSubscriber, trima.TrimaSubscriber, kama.KamaSubscriber,
        bbands.BbandsSubscriber,
        ht_trendline.HtTrendlineSubscriber,

        # # MOMENTUM INDICATORS
        adx.AdxSubscriber,
        # adxr.AdxrSubscriber, apo.ApoSubscriber, aroon.AroonSubscriber, aroonosc.AroonOscSubscriber,
        # bop.BopSubscriber, cci.CciSubscriber, cmo.CmoSubscriber, dx.DxSubscriber,
        macd.MacdSubscriber,
        # # mfi.MfiSubscriber,
        # mom.MomSubscriber, ppo.PpoSubscriber, roc.RocSubscriber, rocr.RocrSubscriber,
        rsi.RsiSubscriber,
        # stoch.StochSubscriber, stochf.StochfSubscriber, stochrsi.StochrsiSubscriber,
        # trix.TrixSubscriber, ultosc.UltoscSubscriber,
        willr.WillrSubscriber, # the last one (if changes, change in SignalSubscriber default subscription)

    ]

def get_doge_subscriber_classes():
    # from apps.doge.doge_TA_actors import DogeSubscriber

    return [
        # place your doges here
        # DogeSubscriber,
    ]
