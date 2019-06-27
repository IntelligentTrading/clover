from django.shortcuts import render
from django.views.generic import ListView
from apps.doge.doge_utils import *
from collections import OrderedDict
import time
from apps.portfolio.models.allocation import Allocation
from django.db.models import F, Func

class AllocationsHistoryView(ListView):

    model = Allocation
    # paginate_by = 10


    def get_allocations(self, delta_time=60*60*8):
        import datetime

        min_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=delta_time)
        allocations = Allocation.objects.annotate(timestamp=Func(F('_timestamp'), function='UNIX_TIMESTAMP')).filter(_timestamp__gte=min_timestamp)
        for allocation in allocations:
            print(allocation)


    def get_queryset(self):
        import datetime
        delta_time = 60*60*24*7
        min_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=delta_time)
        allocations = Allocation.objects.annotate(timestamp=F('_timestamp')).filter(timestamp__gte=min_timestamp)
        return allocations

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        # Add in a QuerySet of all the books
        context['supported_coins'] = ['BTC', 'ETH', 'USDT']
        return context




    # def get(self, request):

    # allocations = self.get_allocations()


    #    context = {
    #        "allocations": allocations,
    #    }

    #    return render(request, 'allocations.html', context)

