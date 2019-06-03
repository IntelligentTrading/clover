import logging

from settings import DEBUG
from settings.redis_db import database
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from apps.TA.storages.data.pv_history import PriceVolumeHistoryStorage, default_price_indexes, default_volume_indexes

logger = logging.getLogger(__name__)


class HistoricalDataAPI(APIView):
    if DEBUG:
        authentication_classes = ()
        permission_classes = ()

    def put(self, request, ticker, format=None):
        """
        This should receive a resampled price
        for the upcoming or nearly past 5min period
        where timestamp is divisible by 300s (5 min)
        and represents a resampled data point for
        :return:
        """

        try:
            database_response = load_ticker(ticker, request.data)

            return Response({
                'success': f'{sum(database_response)} '
                f'db entries created and TA subscribers received'
            }, status=status.HTTP_201_CREATED)


        except Exception as e:
            logger.error(str(e))
            return Response({
                'error': str(e)
            }, status=status.HTTP_501_NOT_IMPLEMENTED)



def load_ticker(ticker, data):
    ticker = ticker or data.get('ticker')
    exchange = data.get('exchange')
    timestamp = data.get('timestamp')

    # SAVE VALUES IN REDIS USING PriceVolumeHistoryStorage OBJECT
    pipeline = database.pipeline()  # transaction=False

    # CREATE OBJECT FOR STORAGE
    data_history = PriceVolumeHistoryStorage(
        ticker=ticker,
        exchange=exchange,
        timestamp=timestamp
    )
    data_history_objects = {}


    if "_ETH" in ticker:
        return Response({
            'success': f'not assessing ETH base tickers, 0 db entries created'
        }, status=status.HTTP_202_ACCEPTED)

    if "_BTC" in ticker:

        close_volume = int(float(data["close_volume"]))
        close_price = int(float(data["close_price"]))

        if close_volume and close_price and (close_volume * close_price) < 50:  # less than 50 BTC volume
            return Response({
                'success': f'volume {close_volume} x {close_price} is low, 0 db entries created'
            }, status=status.HTTP_202_ACCEPTED)

    for index in default_price_indexes + default_volume_indexes:
        if not data.get(index):
            continue

        index_value = float(data[index])

        if index in default_price_indexes:
            index_value = index_value * (10 ** 8)

        index_value = int(index_value)  # if you do int(float()) it will trim values such as 0.1237 to 0 💩

        if index_value > 0:
            data_history.index = index
            data_history.value = int(index_value)

            # ensure the object stays separate in memory
            # (because saving is pipelined)
            data_history_objects[index] = data_history

            # add the saving of this object to the pipeline
            pipeline = data_history_objects[index].save(publish=True, pipeline=pipeline)

    return pipeline.execute()

