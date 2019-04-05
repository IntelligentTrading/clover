from django.shortcuts import render
from django.views.generic import View

from apps.doge.doge_utils import *
from collections import OrderedDict
import time


class CommitteesView(View):

    cached_committees = None

    def dispatch(self, request, *args, **kwargs):

        if CommitteesView.cached_committees is None:
           CommitteesView.cached_committees = load_committees_in_period(ticker='BTC_USDT', exchange='binance', start_timestamp=time.time()-60*60*24*1,
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



    def _get_allocation_info(self, allocation_id):
        from apps.portfolio.models.allocation import Allocation
        result = list(Allocation.objects.filter(id=allocation_id).values())
        return result




    def get(self, request):
        ticker = 'BTC_USDT'

        if not request.path.endswith('/dashboard'):
            ticker = request.path.split('/')[-1]

        committees = load_committees_in_period(ticker=ticker, exchange='binance',
                                               start_timestamp=time.time() - 60*60*24*1,
                                               end_timestamp=time.time())

        # reorder from newest to oldest
        committees.reverse()

        data = []
        for committee in committees:
            committee_data = {'time_str': committee.time_str,
                              'timestamp': committee.timestamp,
                              'benchmark_profit': committee.benchmark_profit,
                              'committee_id': committee.committee_id}
            allocations = committee.get_voted_for_allocations()
            realized_allocations = OrderedDict()
            target_allocations = OrderedDict()
            for allocation_id in allocations:
                # print(self._get_allocation_info(allocation_id))
                # print(self._get_allocation_info(allocation_id)[0]['realized_allocation'])
                # print(self._get_allocation_info(allocation_id)[0]['_timestamp'])
                realized_allocations[self._get_allocation_info(allocation_id)[0]['_timestamp']] = \
                    self._get_allocation_info(allocation_id)[0]['realized_allocation']
                target_allocations[self._get_allocation_info(allocation_id)[0]['_timestamp']] = \
                    self._get_allocation_info(allocation_id)[0]['target_allocation']
            committee_data['realized_allocations'] = realized_allocations
            committee_data['target_allocations'] = target_allocations

            traders = []
            for trader in committee.doge_traders:
                print('Building evaluation object...')
                evaluation = trader.evaluation_object(start_time=committee.start_training_timestamp,
                                         end_time=committee.end_training_timestamp, ticker=ticker)
                # print(evaluation.get_report())
                print('Finished building.')
                trader_data = {}
                performance_dict = trader.performance_at_timestamp(committee.timestamp)
                trader_data['svg'] = self._clean_svg(trader.svg_source_chart)
                trader_data['performance_dict'] = performance_dict
                trader_data['doge_str'] = trader.doge_str
                trader_data['evaluation_report'] = evaluation.get_report().replace('     ', ' ')
                trader_data['buy_and_hold_report'] = evaluation.benchmark_backtest.get_report().replace('     ', ' ')
                traders.append(trader_data)
            committee_data['traders'] = traders
            data.append(committee_data)


        context = {
            "committees": committees,
            "data": data
        }

        return render(request, 'committees.html', context)

