import logging
import time
from datetime import datetime

from django.core.management.base import BaseCommand

from apps.portfolio.models import Portfolio
from apps.portfolio.models.allocation import ITF1HR, ITF6HR, ITF24HR, ITF_PACKS
from apps.portfolio.services.doge_votes import get_allocations_from_doge
from apps.portfolio.services.signals import SHORT_HORIZON, MEDIUM_HORIZON, LONG_HORIZON
from apps.portfolio.services.trading import set_portfolio

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run auto-balancing of managed portfolios'

    def handle(self, *args, **options):
        logger.info("Starting portfolio rebalancer.")

        while True:
            try:
                balance_portfolios()
            except Exception as e:
                logging.critical(str(e))
            time.sleep(60*20)


def balance_portfolios():
    # today = datetime.today()
    # query_time = datetime(today.year, today.month, today.day, today.hour, 20)

    ITF_PACK_HORIZONS = {ITF1HR: SHORT_HORIZON, ITF6HR: MEDIUM_HORIZON, ITF24HR: LONG_HORIZON}



    ITF_doge_binance_allocations, committees_used = \
        get_allocations_from_doge(at_datetime=datetime.now())    # will raise an exception if no votes are found


    # ITF_binance_allocations = {
    #     itf_group: get_allocations_from_signals(horizon=horizon, at_datetime=datetime.now())
    #    for itf_group, horizon in ITF_PACK_HORIZONS.items()
    # }   # disable for now

    for portfolio in Portfolio.objects.all():
        # if portfolio.recently_rebalanced:  # we don't need this logic for Clover so far
        #       continue

        binance_account = portfolio.exchange_accounts.first()

        if not binance_account:
            continue
        if not binance_account.is_active:
            continue

        try:

            '''
            target_allocation = deepcopy(portfolio.target_allocation)
            for alloc in portfolio.target_allocation:
                if alloc['coin'] in ITF_PACKS:
                    itf_pack = alloc['coin']

                    if itf_pack in [ITF1HR, ITF6HR, ITF24HR]:
                        insertion_allocations = ITF_binance_allocations[itf_pack]
                    elif itf_pack == MOONDOGE:
                        insertion_allocations = ITF_doge_binance_allocations
                    elif itf_pack == ITFPRIV:
                        continue  # yet to be implemented
                    else:
                        continue

                    target_allocation = merge_allocations(
                        base_allocation=target_allocation,
                        insert_allocation=insertion_allocations,
                        merge_coin=itf_pack
                    )

            final_target_allocation = clean_allocation(target_allocation)
            '''


            # for now, disable all other options for rebalancing and just use doge allocations
            set_portfolio(portfolio, ITF_doge_binance_allocations, committees_used)  # multithreaded

        except Exception as e:
            logging.error(str(e))
            return str(e)   # for unit tests







def clean_allocation(allocation):

    clean_allocation = []
    portion_sum = 0.0
    for alloc in allocation:
        if alloc['coin'] in ITF_PACKS:
            continue
        if float(alloc['portion']) < 0.0005:
            continue
        if alloc['coin'] == "BTC":
            continue

        portion_sum += float(alloc['portion'])
        clean_allocation.append({
            'coin': alloc['coin'], 'portion': float(alloc['portion']) // 0.0001 / 10000
        })

    clean_allocation.append({
            'coin': "BTC", 'portion': (0.999 - portion_sum)
        })

    return clean_allocation

def merge_allocations(base_allocation, insert_allocation, merge_coin):

    allocation_dict = {alloc['coin']: alloc['portion'] for alloc in base_allocation}
    if merge_coin not in allocation_dict:
        return base_allocation

    portion_multiplier = float(allocation_dict[merge_coin])
    if portion_multiplier < 0.01:
        return base_allocation

    insert_allocation_dict = {alloc['coin']: alloc['portion'] for alloc in insert_allocation}

    for coin, portion in insert_allocation_dict.items():
        if coin not in allocation_dict:
            allocation_dict[coin] = 0.0
        allocation_dict[coin] += portion * portion_multiplier

    return [{"coin":coin, "portion":portion} for coin, portion in allocation_dict.items() if coin != merge_coin]
