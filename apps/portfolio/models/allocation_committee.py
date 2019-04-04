from datetime import timedelta, datetime

from django.contrib.auth import get_user_model
from django.contrib.postgres.fields import JSONField
from django.dispatch import receiver

from apps.portfolio.services.signals import get_BTC_price

User = get_user_model()
from django.db.models.signals import pre_save
from django.db import models


class AllocationCommittee(models.Model):

    allocation = models.ForeignKey(
        'portfolio.Allocation', null=False, on_delete=models.CASCADE
    )

    ticker = models.TextField(max_length=15)
    committee_id = models.TextField(max_length=10)
    voted_at = models.DateTimeField()
    vote = models.FloatField()
    horizon = models.IntegerField()


