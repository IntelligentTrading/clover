from django.db import models
from unixtimestampfield.fields import UnixTimeStampField


METRIC_IDS = {
    'mean_profit': 0,
}
import os.path
BASE = os.path.dirname(os.path.abspath(__file__))

GP_TRAINING_CONFIG = os.path.join(BASE, 'doge_config.json')



class Doge(models.Model):  #todo: can use Timestampable and/or Expirable from common.behaviors
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
