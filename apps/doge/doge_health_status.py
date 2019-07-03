from apps.portfolio.models.allocation import Allocation
from apps.portfolio.models.allocation_committee import AllocationCommittee
import datetime
from django.db.models import F, Func
import logging

from collections import namedtuple

VotesInfo = namedtuple("VotesInfo", "votes earliest latest count committee_ids")

class CloverHealth:

    def __init__(self, past_hours=24):
        min_timestamp = datetime.datetime.now() - datetime.timedelta(hours=past_hours)
        self._allocations = Allocation.objects.annotate(timestamp=F('_timestamp')).filter(timestamp__gte=min_timestamp)
        self._past_hours = past_hours

    @property
    def num_rebalances(self):
        return self._allocations.count()

    @property
    def avg_rebalances_per_hour(self):
        return self.num_rebalances / self._past_hours

    def num_committees_used(self):
        pass  # TODO


    @property
    def latest_allocation(self):
        return self._allocations.latest("timestamp")


    def allocation_votes(self, allocation_id=None):
        if allocation_id == None:
            allocation_id = self.latest_allocation.id

        entries = AllocationCommittee.objects.filter(allocation_id=allocation_id)

        voted_for_tickers = [item["ticker"] for item in entries.values("ticker").distinct()]
        vote_data = {}

        for ticker in voted_for_tickers:

            ticker_votes =  entries.filter(ticker=ticker)
            latest = ticker_votes.latest("voted_at").voted_at
            earliest = ticker_votes.earliest("voted_at").voted_at
            count = ticker_votes.count()
            committee_ids = ticker_votes.values("committee_id").distinct()
            committee_ids = sorted([item["committee_id"] for item in list(committee_ids)])

            info = VotesInfo(votes=ticker_votes, latest=latest, earliest=earliest, count=count, committee_ids=committee_ids)
            vote_data[ticker] = info
            logging.info(f"{ticker} earliest: {earliest}   latest: {latest}    total votes: {count}    committee ids: {committee_ids}")

        return vote_data













