from apps.TA.resources.abstract_subscriber import SubscriberException
from apps.TA.storages.abstract.indicator import IndicatorStorage, BULLISH, BEARISH, OTHER
from apps.TA.storages.abstract.indicator_subscriber import IndicatorSubscriber
from apps.TA.indicators.momentum import willr
from apps.doge.models.doge_train_test import DogeCommittee
from settings import logger, SUPPORTED_DOGE_TICKERS
import time

class SignalSubscriberException(SubscriberException):
    pass


#
class SignalSubscriber(IndicatorSubscriber):
    class_describer = "signal_subscriber"
    classes_subscribing_to = [
        willr.WillrStorage  # the last one
    ]
    storage_class = IndicatorStorage  # override with applicable storage class



class DogeVoteStorage(IndicatorStorage):
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
    storage_class = DogeVoteStorage  # override with applicable storage class

    def __init__(self, *args, **kwargs):
        self._reload_committee()
        super().__init__(*args, **kwargs)
        logger.info("                                                      (😎 IT IS THE LAST ONE 😎)")
        logger.info(f'Initialized DogeSubscriber at {time.time()}')

    def _reload_committee(self):
        self.committee = DogeCommittee()

    def handle(self, channel, data, *args, **kwargs):
        # check if we received data for a ticker we support
        if self.ticker not in SUPPORTED_DOGE_TICKERS:  # @tomcounsell please check if this is OK or I should register
                                                       # for tickers of interest in some other way
            logger.debug(f'Ticker {self.ticker} not in {SUPPORTED_DOGE_TICKERS}, skipping...')
            return

        # check if the committee has expired
        if self.committee.expired:
            logger.info('Doge committee expired, reloading...')
            self._reload_committee()

        logger.info(f'Doge subscriber invoked at {self.timestamp}, channel={str(channel)}, data={str(data)} '
                    f'(it is now {time.time()})')
        transaction_currency, counter_currency = self.ticker.split('_')

        new_doge_storage = DogeVoteStorage(ticker=self.ticker,
                                           exchange=self.exchange,
                                           timestamp=self.timestamp,
                                           periods=self.committee.periods)

        ticker_votes, weights = self.committee.vote(transaction_currency, counter_currency, self.timestamp)
        # weighted_vote = sum([ticker_votes[i] * weights[i] for i in range(len(ticker_votes))]) / sum(weights)

        new_doge_storage.value = (sum(ticker_votes) * 100 / len(ticker_votes))  # normalize to +-100 scale
        new_doge_storage.save(publish=True)
        logger.info('Doge vote saved')


    def pre_handle(self, channel, data, *args, **kwargs):
        super().pre_handle(channel, data, *args, **kwargs)
