from apps.TA.resources.abstract_subscriber import SubscriberException
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH, OTHER
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.indicators.momentum import willr
from apps.TA.storages.abstract.key_value import KeyValueStorage
from apps.TA.storages.abstract.ticker import TickerStorage


class SignalSubscriberException(SubscriberException):
    pass


class SignalSubscriber(IndicatorSubscriber):
    class_describer = "signal_subscriber"
    classes_subscribing_to = [
        willr.WillrStorage  # the last one
    ]
    storage_class = IndicatorStorage  # override with applicable storage class


class DogeStorage(KeyValueStorage):
    HASH_DIGITS_TO_KEEP = 8

    # self.value = string_format_of_entire_decision_tree
    # self.db_key_prefix = "ticker:exchange" # but we don't care, so don't distinguish!
    # self.db_key_suffix = str(hash(self.value))[:8] #last 8 chars of the hash

    @staticmethod
    def hash(doge_str):
        return hash(doge_str) % 10 ** DogeStorage.HASH_DIGITS_TO_KEEP

    @staticmethod
    def get_doge_str(doge_hash):
        doge_storage = DogeStorage(key_suffix=doge_hash)
        return doge_storage.get_value().decode('utf8')


class DogePerformance(TickerStorage):
    """
        defines the performance score for a doge over time, unique per ticker
    """
    # self.key_suffix = doge_id
    #
    # self.ticker = ticker
    # self.exchange = exchange
    # self.value = performance_score
    # self.timestamp


    @staticmethod
    def weight_at_timestamp(doge_id, ticker, exchange, timestamp, metric_id=0):
        result = DogePerformance.query(key_suffix=f'{doge_id}:{metric_id}',
                                       ticker=ticker,
                                       exchange=exchange,
                                       timestamp=timestamp)
        return float(result['values'][-1])


class CommitteeStorage(TickerStorage):
    """
        defines which doges are valid for voting in the committee at the timestamp
    """
    #
    # self.ticker = ticker
    # self.exchange = exchange
    # self.value = str(doge_id_list)
    # self.timestamp = timestamp

    @staticmethod
    def load_rockstars(num_previous_committees_to_search, num_rockstars,
                       ticker, exchange, timestamp):
        committees = CommitteeStorage.query(ticker=ticker, exchange=exchange, timestamp=timestamp,
                                            timestamp_tolerance=30000 * num_previous_committees_to_search)
        doge_ids = []
        weights = []
        for committee_ids, score in zip(committees['values'], committees['scores']):
            timestamp = CommitteeStorage.timestamp_from_score(score)
            committee_ids = committee_ids.split(':')
            for doge_id in committee_ids:
                weight = DogePerformance.weight_at_timestamp(doge_id, ticker, exchange, timestamp)
                doge_ids.append(doge_id)
                weights.append(weight)
        doge_ids = [doge_id for _, doge_id in sorted(zip(weights, doge_ids))]
        rockstar_ids = doge_ids[:num_rockstars]
        doge_strs = [DogeStorage.get_doge_str(doge_hash) for doge_hash in doge_ids]
        return doge_strs





class CommitteeVoteStorage(IndicatorStorage):
    # todo: abstract this for programatic implementation
    requisite_TA_storages = ["rsi", "sma"]  # example

    def produce_signal(self):
        if self.vote_trend and abs(self.vote) >= 25:
            self.send_signal(trend=self.vote_trend)


    @property
    def vote_trend(self):
        if not self.value:
            return None

        if self.value > 0:
            return BULLISH
        elif self.value < 0:
            return BEARISH
        else:
            return OTHER

    @property
    def vote(self):
        return self.value


def clean_redis():
    from settings.redis_db import database
    for key in database.keys('*Doge*'):
        database.delete(key)
    for key in database.keys('*Committee*'):
        database.delete(key)

