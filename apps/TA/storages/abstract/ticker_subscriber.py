import json
from abc import ABC
import logging
from json import JSONDecodeError

from apps.TA import TAException
from apps.TA.resources.abstract_subscriber import SubscriberException, AbstractSubscriber

logger = logging.getLogger(__name__)


class TickerSubscriber(AbstractSubscriber):
    class_describer = "ticker_subscriber"
    classes_subscribing_to = [
        # ...
    ]

def timestamp_is_near_5min(timestamp) -> bool:
    # close to a five minute period mark? (+ or - 45 seconds)
    seconds_from_five_min = (int(timestamp) + 45) % 300
    return bool(seconds_from_five_min < 90)

def score_is_near_5min(score) -> bool:
    return bool(round(score) - 45/300 < score < round(score) + 45/300)

def get_nearest_5min_timestamp(timestamp) -> int:
    return ((int(timestamp) + 45) // 300) * 300

def get_nearest_5min_score(score) -> int:
    return int(round(score))
