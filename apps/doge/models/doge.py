import logging

from django.contrib.auth import get_user_model
from django.db import models
from unixtimestampfield.fields import UnixTimeStampField
# from apps.common.behaviors import Timestampable TODO: @tomcounsell do we need this?

from apps.genetic_algorithms.gp_artemis import ExperimentManager
from apps.backtesting.data_sources import redis_db
from apps.genetic_algorithms.genetic_program import GeneticTickerStrategy
from apps.backtesting.tick_provider import PriceDataframeTickProvider
from apps.backtesting.tick_listener import TickListener
from apps.backtesting.utils import datetime_from_timestamp
from apps.genetic_algorithms.gp_utils import Period
from apps.genetic_algorithms.leaf_functions import RedisDummyTAProvider

METRIC_IDS = {
    'mean_profit': 0,
}

GP_TRAINING_CONFIG = 'doge_config.json'


class Doge(models.Model):
    train_start_timestamp = UnixTimeStampField()    # timestamp of the first data point in the training dataset
    train_end_timestamp = UnixTimeStampField()      # timestamp of the last data point in the training dataset
    experiment_id = models.TextField()              # a string representation of the experiment showing the tickers
                                                    # in the training set, start and end times and other parameters
                                                    # (used by Artemis experiment management library)
    rank = models.IntegerField()                    # the rank of the doge in its batch (lower is better)
    representation = models.TextField()             # a textual representation of the doge strategy
    metric_id = models.SmallIntegerField()          # performance metric id (e.g. 0=mean profit, TODO)
    metric_value = models.FloatField()              # value of the metric

    @staticmethod
    def _create_instance_and_write(train_start_timestamp, train_end_timestamp, experiment_id, rank, representation,
                                   metric_id, metric_value):

        doge = Doge()
        doge.train_start_timestamp = train_start_timestamp
        doge.train_end_timestamp = train_end_timestamp
        doge.experiment_id = experiment_id
        doge.rank = rank
        doge.representation = representation
        doge.metric_id = metric_id
        doge.metric_value = metric_value
        doge.save()


class DogeTrainer:

    def __init__(self, database):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()
        self.database = database

    def retrain_doges(self, start_timestamp, end_timestamp, max_doges_to_save=10):
        config_json = self.fill_json_template(self.gp_training_config_json, start_timestamp, end_timestamp)

        logging.info('>>>>>>> Starting GP training... ')
        logging.info(f'    >>> start_time = {datetime_from_timestamp(start_timestamp)}')
        logging.info(f'    >>> end_time = {datetime_from_timestamp(end_timestamp)}')

        # create an experiment manager
        e = ExperimentManager(experiment_container=config_json, read_from_file=False, database=self.database,
                              hof_size=10)  # we will have one central json with all the parameters

        # run experiments
        e.run_experiments(keep_record=True) # if we can run it in parallel, otherwise call e.run_experiments()

        logging.info('>>>>>>> GP training completed.')

        # retrieve best performing doges
        logging.info('>>>>>>> Ranking doges by performance...')
        doge_df = e.get_best_performing_across_variants_and_datasets(datasets=e.training_data,
                                                                     sort_by=['mean_profit'],
                                                                     top_n_per_variant=1)
        logging.info('>>>>>>> Ranking completed.')

        # write these doges to database
        logging.info('>>>>>>> Saving GPs to database...')

        for i, row in enumerate(doge_df.itertuples()):
            if i > max_doges_to_save:
                break
            # save experiment id and doge
            Doge._create_instance_and_write(train_start_timestamp=start_timestamp,
                                            train_end_timestamp=end_timestamp,
                                            experiment_id=row.variant.name,
                                            rank=i,
                                            representation=str(row.doge),
                                            metric_id=METRIC_IDS['mean_profit'],
                                            metric_value=row.mean_profit)

        logging.info('>>>>>>> GPs saved to database.')


    @staticmethod
    def fill_json_template(gp_training_config_json, start_timestamp, end_timestamp):
        return gp_training_config_json.format(
            start_time=datetime_from_timestamp(start_timestamp),
            end_time=datetime_from_timestamp(end_timestamp)
        )

    @staticmethod
    def run_training(start_timestamp, end_timestamp):
        # TODO: replace with datetime.now() and similar beautiful stuff
        training_period = Period('2018/04/01 00:00:00 UTC', '2018/04/08 00:00:00 UTC')
        trainer = DogeTrainer(redis_db)
        trainer.retrain_doges(training_period.start_time, training_period.end_time, 10)

        #trader = DogeTrader(database=redis_db)


class DogeTrader(TickListener):

    def __init__(self, database=redis_db):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()

        self.doge_strategies = self._load_latest_doge_strategies(database)

        # dummy tick provider for now, TODO: replace with actual one
        e = ExperimentManager('gv5_experiments.json', database=database)
        tick_provider = PriceDataframeTickProvider(e.training_data[0].price_data)
        tick_provider.add_listener(self)
        tick_provider.run()

        # simulate the decision process over all the strategies

    def _load_latest_doge_strategies(self, database):
        doge_strategies = []
        transaction_currency = 'BTC'
        counter_currency = 'USDT'
        resample_period = 60
        source = 0
        function_provider = RedisDummyTAProvider()

        # get doges out of DB
        last_timestamp = Doge.objects.latest('train_end_timestamp').train_end_timestamp.timestamp()  # ah well :)
        dogi = Doge.objects.filter(train_end_timestamp=last_timestamp)

        for doge_object in dogi:
            start_timestamp = doge_object.train_start_timestamp.timestamp()
            end_timestamp = doge_object.train_end_timestamp.timestamp()
            individual_str = doge_object.representation
            experiment_id = doge_object.experiment_id

            experiment_json = DogeTrainer.fill_json_template(self.gp_training_config_json,
                                                             int(float(start_timestamp)), int(float(end_timestamp)))
            doge, gp = ExperimentManager.resurrect_doge(experiment_json, experiment_id, individual_str, database,
                                                        function_provider)
            strategy = GeneticTickerStrategy(doge, transaction_currency, counter_currency, resample_period,
                                             source, gp)
            doge_strategies.append(strategy)

        return doge_strategies

    def process_event(self, price_data, signal_data):
        print(f'So wow! Price arrived ({datetime_from_timestamp(price_data.Index)})')
        for i, doge in enumerate(self.doge_strategies):
            print(f'  Doge {i} says: {str(doge.process_ticker(price_data, signal_data).outcome)}')


    def broadcast_ended(self):
        print('Doges have spoken.')



# autotrading:
# get the most recent doge
#


if __name__ == '__main__':
    training_period = Period('2018/04/01 00:00:00 UTC', '2018/04/08 00:00:00 UTC')
    trainer = DogeTrainer(redis_db)
    trainer.retrain_doges(training_period.start_time, training_period.end_time, 10)
    trader = DogeTrader(database=redis_db)

