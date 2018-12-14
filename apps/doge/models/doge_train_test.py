import logging

from apps.backtesting.data_sources import redis_db, postgres_db
from apps.backtesting.tick_listener import TickListener
from apps.backtesting.tick_provider_heartbeat import TickProviderHeartbeat
from apps.backtesting.tick_provider_itf_db import TickProviderITFDB
from apps.backtesting.tick_provider import PriceDataframeTickProvider
from apps.backtesting.utils import datetime_from_timestamp
from apps.doge.models import Doge
from apps.doge.models.doge import GP_TRAINING_CONFIG, METRIC_IDS
from apps.genetic_algorithms.genetic_program import GeneticTickerStrategy
from apps.genetic_algorithms.gp_artemis import ExperimentManager
from apps.genetic_algorithms.gp_utils import Period
from apps.genetic_algorithms.leaf_functions import RedisTAProvider



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
        training_period = Period('2018/02/01 00:00:00 UTC', '2018/02/07 00:00:00 UTC')
        trainer = DogeTrainer(redis_db)
        start_time = redis_db.get_nearest_db_timestamp(training_period.start_time, 'BTC', 'USDT', None, None)
        end_time = redis_db.get_nearest_db_timestamp(training_period.end_time, 'BTC', 'USDT', None, None)
        trainer.retrain_doges(start_time, end_time, 10)

        #trader = DogeTrader(database=redis_db)


class DogeTrader:

    def __init__(self, database, doge_object, function_provider, gp_training_config_json):
        self.train_start_timestamp = doge_object.train_start_timestamp.timestamp()
        self.train_end_timestamp = doge_object.train_end_timestamp.timestamp()
        self.individual_str = doge_object.representation
        self.experiment_id = doge_object.experiment_id
        self.metric_id = doge_object.metric_id
        self.metric_value = doge_object.metric_value
        self.gp_training_config_json = gp_training_config_json
        experiment_json = DogeTrainer.fill_json_template(self.gp_training_config_json,
                                                         int(float(self.train_start_timestamp)),
                                                         int(float(self.train_end_timestamp)))
        self.doge, self.gp = ExperimentManager.resurrect_doge(experiment_json, self.experiment_id, self.individual_str,
                                                              database, function_provider)

        self.strategy = GeneticTickerStrategy(tree=self.doge, gp_object=self.gp)

    def vote(self, ticker_data):
        return self.strategy.process_ticker(ticker_data)



class DogeTradingManager(TickListener):

    def __init__(self, database=redis_db, heartbeat_period_secs=60):
        with open(GP_TRAINING_CONFIG, 'r') as f:
            self.gp_training_config_json = f.read()

        self.function_provider = RedisTAProvider()
        self.doge_strategies = self._load_latest_doge_strategies(database)

        # dummy tick provider for now, TODO: replace with actual one
        # e = ExperimentManager('gv5_experiments.json', database=database)
        # tick_provider = PriceDataframeTickProvider(e.training_data[0].price_data,
        #                                           transaction_currency=e.training_data[0].transaction_currency,
        #                                           counter_currency=e.training_data[0].counter_currency,
        #                                           source=e.training_data[0].source,
        #                                           resample_period=e.training_data[0].resample_period)
        # tick_provider.add_listener(self)
        # tick_provider.run()


        tick_provider_heartbeat = TickProviderHeartbeat(
            heartbeat_period_secs=heartbeat_period_secs,
            database=database,
            ticker_list=['BTC_USDT', 'ETH_BTC', 'OMG_BTC']
        )
        tick_provider_heartbeat.add_listener(self)
        tick_provider_heartbeat.run()

    def _load_latest_doge_strategies(self, database):
        doge_traders = []

        # get doges out of DB
        last_timestamp = Doge.objects.latest('train_end_timestamp').train_end_timestamp.timestamp()  # ah well :)
        dogi = Doge.objects.filter(train_end_timestamp=last_timestamp)

        for doge_object in dogi:
            doge = DogeTrader(database=database, doge_object=doge_object, function_provider=self.function_provider,
                              gp_training_config_json=self.gp_training_config_json)
            doge_traders.append(doge)

        return doge_traders

    def process_event(self, ticker_data):
        """
        This method will be called every heartbeat_period_secs seconds for all the tickers on which the trader
        is subscribed.
        :param ticker_data: an instance of TickerData (OHLCV for a ticker at timestamp)
        :return:
        """
        print(f'So wow! Price for {ticker_data.transaction_currency}-{ticker_data.counter_currency} '
              f'arrived ({datetime_from_timestamp(ticker_data.timestamp)})')

        sum_votes = total_weight = 0
        for i, doge in enumerate(self.doge_strategies):
            decision = doge.vote(ticker_data)
            print(f'  Doge {i} says: {str(decision)} (its weight is {doge.metric_value})')
            sum_votes += decision.outcome * doge.metric_value  # outcome is +1 for buy, -1 for sell, 0 for ignore
            total_weight += doge.metric_value

        print(f'Total sum of votes: {sum_votes}')
        print(f'Total doge weight: {total_weight}')
        print(f'End decision for {ticker_data.transaction_currency}-{ticker_data.counter_currency} '
              f'at {datetime_from_timestamp(ticker_data.timestamp)}: {sum_votes/total_weight}\n\n')


    def broadcast_ended(self):
        print('Doges have spoken.')