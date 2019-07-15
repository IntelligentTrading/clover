from django.shortcuts import render
from django.views.generic import View

from apps.doge.doge_utils import *
from apps.doge.doge_health_status import CloverHealth
from collections import OrderedDict
import time


class DogeHealthView(View):

    def get(self, request, hours=4):

        health_info = CloverHealth(past_hours=int(hours))

        context = {
            "health_info": health_info,
            "hours": hours,
            "latest_allocation_votes": health_info.allocation_votes()
        }

        return render(request, 'health_status.html', context)

