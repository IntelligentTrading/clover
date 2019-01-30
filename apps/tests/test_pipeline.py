from settings import logger
from settings.redis_db import database
from datetime import datetime
from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
from settings.redis_db import database as db
from apps.TA import PRICE_INDEXES, VOLUME_INDEXES, JAN_1_2017_TIMESTAMP
from apps.TA.indicators.overlap import sma, ema, wma, dema, tema, trima, bbands, ht_trendline, kama, midprice
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.storages.data.price import PriceStorage
from django.test import TestCase


ticker = "CWC_ETH"
exchange = "test"
index = "close_price"
periods = 5
timestamp = datetime(2018,12,15).timestamp()
score = TimeseriesStorage.score_from_timestamp(timestamp)
key = f"{ticker}:{exchange}:PriceStorage:{index}"


class IndicatorsTestCase(TestCase):

    def setUp(self):

        results = PriceStorage.query(
            ticker=ticker,
            exchange=exchange,
            index=index,
            periods_range=periods
        )


    def test_score(self):
        value = db.zrangebyscore(key, score, score)[0].decode("utf-8").rsplit(":", 1)[0]

        print("the value", value, "the score", score)

        price = PriceStorage(ticker=ticker, exchange=exchange, index=index, timestamp=timestamp)
        price.value = float(value)+1
        price.save()

        value = db.zrangebyscore(key, score, score)[0].decode("utf-8").rsplit(":", 1)[0]
        print("the new value", value, "the score", score)

        db_records = db.zrangebyscore(key, score, score)
        print("db records: ", db_records)


