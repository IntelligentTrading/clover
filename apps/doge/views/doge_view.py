from copy import deepcopy

from django.contrib import messages
from django.shortcuts import render, redirect
from django.views.generic import View

from apps.portfolio.models.allocation import ITF_PACKS
from apps.portfolio.services.binance import binance_coins

from apps.doge.doge_utils import *
import time



class CommitteesView(View):

    cached_committees = None

    def dispatch(self, request, *args, **kwargs):

        if CommitteesView.cached_committees is None:
           CommitteesView.cached_committees = load_committees_in_period(ticker='BTC_USDT', exchange='binance', start_timestamp=time.time()-60*60*24,
                                               end_timestamp=time.time())
        print(f'Loaded {len(CommitteesView.cached_committees)} committees.')


        return super().dispatch(request, *args, **kwargs)

    def get(self, request):

        data = []
        for committee in CommitteesView.cached_committees:
            committee_data = {'timestamp': committee.timestamp}
            traders = []
            for trader in committee.doge_traders:
                trader_data = {}
                trader_data['svg'] = trader.svg_source_chart
                trader_data['weight_at_timestamp'] = trader.weight_at_timestamp(committee.timestamp)
                traders.append(trader)
            committee_data['traders'] = traders
            data.append(committee_data)



        context = {
            "committees": CommitteesView.cached_committees,
            "data": data
        }

        return render(request, 'committees.html', context)

