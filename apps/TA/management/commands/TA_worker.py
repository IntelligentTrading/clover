import logging
import time

from django.core.management.base import BaseCommand

from apps.TA.indicators.fantasy import dracarys
from settings.redis_db import database

logger = logging.getLogger(__name__)

try:
    earliest_price_score = int(float(database.zrangebyscore("BTC_USDT:binance:PriceStorage:close_price", 0, "inf", 0, 1)[0].decode("utf-8").split(":")[0]))
except:
    from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
    earliest_price_score = TimeseriesStorage.score_from_timestamp(int(time.time()))


# TODO:
# Send signals to SNS
# Confirm TA signals are same as old
# Fully replace old signals with new
# Stop using old Aurora database
# turn signals into votes for portfolio


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
            start = time.time()

            logger.info('Restarting TA computation...')

            for class_name in subscribers:
                # logger.debug(f'checking subscription {class_name}: {subscribers[class_name]}')

                try:
                    # logger.debug("calling "+class_name)
                    subscribers[class_name]()  # run subscriber class

                except Exception as e:
                    logger.error(str(e))
                    logger.debug(subscribers[class_name].__dict__)

                time.sleep(1)

            end = time.time()
            logger.info(f'!!! Running all subscribers took {(end-start)/60:.2f} minutes. !!!')

            time.sleep(120)  # be nice to the system :)



def get_subscriber_classes():

    from apps.TA.storages.data.price import PriceSubscriber
    # from apps.TA.storages.data.volume import VolumeSubscriber
    # only PriceStorage:close_price is publishing. All other p and v indexes are muted

    from apps.TA.indicators.overlap import sma, ema, wma, dema, tema, trima, bbands, ht_trendline, kama, midprice
    from apps.TA.indicators.momentum import adx, adxr, apo, aroon, aroonosc, bop, cci, cmo, dx, macd, mom, ppo, \
        roc, rocr, rsi, stoch, stochf, stochrsi, trix, ultosc, willr
    from apps.TA.indicators.events import bbands_squeeze_180min

    return [

        PriceSubscriber,
        # VolumeSubscriber,  # the PriceSubscriber handles volume resampling

        # OVERLAP INDICATORS
        # midprice.MidpriceSubscriber,
        sma.SmaSubscriber, ema.EmaSubscriber, wma.WmaSubscriber,
        # dema.DemaSubscriber, tema.TemaSubscriber, trima.TrimaSubscriber, kama.KamaSubscriber,
        bbands.BbandsSubscriber,
        # ht_trendline.HtTrendlineSubscriber,

        # MOMENTUM INDICATORS
        # adx.AdxSubscriber,
        # adxr.AdxrSubscriber, apo.ApoSubscriber, aroon.AroonSubscriber, aroonosc.AroonOscSubscriber,
        # bop.BopSubscriber, cci.CciSubscriber, cmo.CmoSubscriber, dx.DxSubscriber,
        # macd.MacdSubscriber,
        # # mfi.MfiSubscriber,
        # mom.MomSubscriber, ppo.PpoSubscriber, roc.RocSubscriber, rocr.RocrSubscriber,
        rsi.RsiSubscriber,
        # stoch.StochSubscriber,
        # stochf.StochfSubscriber, stochrsi.StochrsiSubscriber,
        # trix.TrixSubscriber, ultosc.UltoscSubscriber,
        willr.WillrSubscriber,  # formerly know as "THE LAST ONE"

        # EVENTS INDICATORS
        bbands_squeeze_180min.BbandsSqueeze180MinSubscriber,

        # 🔥🔥🔥🔥🔥🐉🔥🔥🔥🔥🔥
        dracarys.DracarysSubscriber,  # the last one (if changes, change in SignalSubscriber default subscription)

    ]

def get_doge_subscriber_classes():
    from apps.doge.doge_train_test import DogeSubscriber

    return [
        DogeSubscriber,
    ]
