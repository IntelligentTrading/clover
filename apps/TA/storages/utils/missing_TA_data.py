from apps.TA.management.commands.TA_worker import get_subscriber_classes
from apps.TA.storages.abstract.timeseries_storage import TimeseriesStorage
from apps.TA.storages.utils.list_search import missing_elements
import logging
from datetime import datetime


logger = logging.getLogger(__name__)

def fill_TA_storage_data_gaps(ticker, exchange, storage_class_name):
    remaining_missing_scores = []

    valid_storage_classes = get_subscriber_classes()

    storage_class = next(
        (
            storage_class for storage_class in valid_storage_classes
            if storage_class.__name__ == storage_class_name
        ),
        None
    )

    if not storage_class:
        return remaining_missing_scores

    for periods in storage_class.get_periods_list():

        now_score = int(TimeseriesStorage.score_from_timestamp(datetime.now().timestamp()))
        one_month = 1*30*24*12  # number of 5 min periods
        query_results = storage_class.query(ticker=ticker, exchange=exchange, periods_key=periods, periods_range=one_month)
        scores = [int(float(score)) for score in query_results['scores']]
        scores.append(now_score)
        scores.append(now_score-one_month)

        missing_scores = missing_elements(scores)

        logger.info(f"{len(missing_scores)} total scores are missing. start filling...")

        for score in missing_scores:
            timestamp = TimeseriesStorage.timestamp_from_score(score)

            storage_object = storage_class(ticker=ticker, exchange=exchange, timestamp=timestamp, periods=periods)
            value = storage_object.get_value()
            if not value:
                remaining_missing_scores.append(score)
            else:
                logger.debug(f"filled one for {storage_class} at score {score}")

    return remaining_missing_scores
