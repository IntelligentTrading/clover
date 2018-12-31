from apps.TA.resources.abstract_subscriber import SubscriberException
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH, OTHER
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.indicators.momentum import willr
from apps.doge.models.doge_train_test import DogeCommittee
from settings import logger


class SignalSubscriberException(SubscriberException):
    pass


#
class SignalSubscriber(IndicatorSubscriber):
    class_describer = "signal_subscriber"
    classes_subscribing_to = [
        willr.WillrStorage  # the last one
    ]
    storage_class = IndicatorStorage  # override with applicable storage class


class DogeStorage(IndicatorStorage):
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


class DogeSubscriber(SignalSubscriber):
    storage_class = DogeStorage  # override with applicable storage class

    def __init__(self, *args, **kwargs):
        self.committee = DogeCommittee()
        # todo: set an expiry for the committee on the same schedule as training
        # todo: ideally, a new training would expire all previous committees, see common.behaviours.expirable
        super().__init__(*args, **kwargs)

    def handle(self, channel, data, *args, **kwargs):
        transaction_currency, counter_currency = self.ticker.split('_')

        new_doge_storage = DogeStorage(ticker=self.ticker,
                                       exchange=self.exchange,
                                       timestamp=self.timestamp, )

        ticker_votes, weights = self.committee.vote(transaction_currency, counter_currency)
        # weighted_vote = sum([ticker_votes[i] * weights[i] for i in range(len(ticker_votes))]) / sum(weights)

        new_doge_storage.value = (sum(ticker_votes) * 100 / len(ticker_votes))  # normalize to +-100 scale
        new_doge_storage.save(publish=True)


    def pre_handle(self, channel, data, *args, **kwargs):
        super().pre_handle(channel, data, *args, **kwargs)
