import pandas as pd
from apps.backtesting.data_sources import db_interface
from apps.backtesting.signals import Signal
from apps.backtesting.tick_provider import TickProvider, TickerData


class TickProviderITFDB(TickProvider):

    def __init__(self, transaction_currency, counter_currency, start_time, end_time, source=0,
                 resample_period=60, database=db_interface):

        super().__init__()
        self.database = database
        signals_df = self.database.get_filtered_signals(start_time=start_time,
                                                      end_time=end_time,
                                                      counter_currency=counter_currency,
                                                      transaction_currency=transaction_currency,
                                                      resample_period=resample_period,
                                                      source=source,
                                                      return_df=True)
        self.prices_df = self.database.get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency, resample_period)

        # move timestamp to column
        self.prices_df.reset_index(level=0, inplace=True)
        signals_df.reset_index(level=0, inplace=True)
        signals_df = signals_df.dropna()
        self.prices_df = self.prices_df.dropna().sort_values(['timestamp'])  # something wrong with the DB!

        # need to merge signal and price timestamps, as they don't match in the database
        self.reindexed_signals_df = pd.merge_asof(signals_df.sort_values('timestamp'),
                                                  self.prices_df,
                                                  direction='nearest')
        self.reindexed_signals_df.sort_values(['timestamp'])


    def run(self):
        for i, price_data in self.prices_df.iterrows():
            signals_now = self.reindexed_signals_df[self.reindexed_signals_df['timestamp'] == price_data['timestamp']]
            signals_now = Signal.pandas_to_objects_list(signals_now)

            self.notify_listeners(price_data, signals_now)
        self.broadcast_ended()