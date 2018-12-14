from apps.TA.resources.abstract_subscriber import SubscriberException
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.indicators.momentum import willr


class SignalSubscriberException(SubscriberException):
    pass


class SignalSubscriber(IndicatorSubscriber):
    class_describer = "signal_subscriber"
    classes_subscribing_to = [
        willr.WillrStorage # the last one
    ]
    storage_class = IndicatorStorage  # override with applicable storage class


    def pre_handle(self, channel, data, *args, **kwargs):
        super().pre_handle(channel, data, *args, **kwargs)



class DogeSubscriber(SignalSubscriber):
    class_describer = "signal_subscriber"
    classes_subscribing_to = [
        willr.WillrStorage # the last one
    ]
    storage_class = IndicatorStorage  # override with applicable storage class

    # estimate when TA should be finished
    # activate the doges 🐕🐕🐕
    # votes = {}
    # for ticker in ['BTC_USDT']:
    #     votes['BTC_USDT'] = sum(doge.vote("BTC_USDT") * doge.weight for doge in living_doges)


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
