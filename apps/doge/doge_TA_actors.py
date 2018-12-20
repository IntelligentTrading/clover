from apps.TA.resources.abstract_subscriber import SubscriberException
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH
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
        willr.WillrStorage # the last one
    ]
    storage_class = IndicatorStorage  # override with applicable storage class


class DogeStorage(IndicatorStorage):

    # todo: abstract this for programatic implementation
    requisite_TA_storages = ["rsi", "sma"]  # example


    def vote(self, ticker):  # 🐕

        # todo: load decision tree
        # todo: query for TA values
        # todo: compute decision tree

        if True:
            return BULLISH
        else:
            return BEARISH


class DogeSubscriber(SignalSubscriber):

    storage_class = DogeStorage  # override with applicable storage class

    def handle(self, channel, data, *args, **kwargs):

        self.index = self.key_suffix

        if str(self.index) is not "close_price":
            logger.debug(f'index {self.index} is not close_price ...ignoring...')
            return

        # get current votes
        committee = DogeCommittee()  # TODO: create only once and reuse

        transaction_currency, counter_currency = self.ticker.split('_')
        ticker_votes, weights = committee.vote(transaction_currency, counter_currency)
        weighted_vote = sum([ticker_votes[i]*weights[i] for i in range(len(ticker_votes))]) / sum(weights)

        new_doge_storage = DogeStorage(ticker=self.ticker,
                                         exchange=self.exchange,
                                         timestamp=self.timestamp,)
        new_doge_storage.value = weighted_vote
        new_doge_storage.save(publish=True)


    # estimate when TA should be finished
    # activate the doges 🐕🐕🐕
    # votes = {}
    # for ticker in ['BTC_USDT']:
    #     votes['BTC_USDT'] = sum(doge.vote("BTC_USDT") * doge.weight for doge in living_doges)

    def pre_handle(self, channel, data, *args, **kwargs):
        super().pre_handle(channel, data, *args, **kwargs)
