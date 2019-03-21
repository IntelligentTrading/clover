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
           CommitteesView.cached_committees = load_committees_in_period(ticker='BTC_USDT', exchange='binance', start_timestamp=time.time()-60*60*24*4,
                                               end_timestamp=time.time())
        print(f'Loaded {len(CommitteesView.cached_committees)} committees.')


        return super().dispatch(request, *args, **kwargs)


    def _clean_svg(self, svg):
        lines = []
        for line in svg.split('\n'):
            if line.strip().startswith('<svg'):
                line = line.split()
                elements = []
                for x in line:
                    if 'width' in x:
                        elements.append('width="100%"')
                    elif 'height' in x:
                        elements.append('height="100%"')
                    else:
                        elements.append(x)
                # line = " ".join([x for x in line if not 'width' in x and not 'height' in x])
                lines.append(" ".join(elements))
                continue
            if 'viewBox' in line:  # remove viewBox
                # opening_quote_index = line.index('viewBox="') + len('viewBox="') - 1
                # closing_quote_index = line[opening_quote_index+1:].index('"') + opening_quote_index
                # line = line[:line.index('viewBox=')] + line[closing_quote_index+1:]
                lines.append(line)
                continue
            elif line.strip().startswith('<polygon fill="white"'):   # remove white background
                continue
            else:
                line = line.replace('fill="#333333"> yes',
                                    'fill="white" > yes')

                line = line.replace('fill="#333333"> &#160;&#160;&#160;no',
                                    'fill="white"> &#160;&#160;&#160;no')

                lines.append(line)
        return '\n'.join(lines)


    def get(self, request):

        data = []
        for committee in CommitteesView.cached_committees:
            committee_data = {'time_str': committee.time_str}
            traders = []
            for trader in committee.doge_traders:
                trader_data = {}
                trader_data['svg'] = self._clean_svg(trader.svg_source_chart)
                trader_data['weight_at_timestamp'] = trader.weight_at_timestamp(committee.timestamp)
                traders.append(trader_data)
            committee_data['traders'] = traders
            data.append(committee_data)



        context = {
            "committees": CommitteesView.cached_committees,
            "data": data
        }

        return render(request, 'committees.html', context)

