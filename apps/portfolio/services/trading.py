import json
import time
from copy import deepcopy
from datetime import datetime

import requests
import logging

from apps.common.utilities.multithreading import start_new_thread
from apps.portfolio.services.binance import translate_allocs_binance_coins, reverse_translate_allocs_binance_coins
from apps.portfolio.services.signals import get_BTC_price
from settings import ITF_TRADING_API_URL, ITF_TRADING_API_KEY, DEBUG
from apps.backtesting.utils import datetime_from_timestamp

class TradingAPIException(Exception):
    pass

def get_binance_portfolio_data(binance_account):

    api_url_base = ITF_TRADING_API_URL
    api_url = api_url_base + "portfolio/"
    headers = {'Content-Type': 'application/json'}
    data = {
        "api_key": ITF_TRADING_API_KEY,
        "binance": {
            "api_key": binance_account.api_key,
            "secret_key": binance_account.secret_key
        }
    }

    if DEBUG:
        logging.debug(str(data))

    response = requests.post(api_url, json=data, headers=headers)
    logging.debug(response.text)

    # https://github.com/IntelligentTrading/trading#binance-exchange-get-portfolio-state
    # if response.status_code == 400:
    #     return response.json()['detail']
    #
    # if response.status_code == 200:
    #     return response.json()['binance']

    return (response.status_code, response.json())


#@start_new_thread
def set_portfolio(portfolio, allocation, committees_used):
    logging.critical(f'set_portfolio called with allocation {str(allocation)}, committees used:  {str(committees_used)}')
    binance_account = portfolio.exchange_accounts.first()
    if not binance_account:
        return
    if not binance_account.is_active:
        return

    allocation = translate_allocs_binance_coins(allocation)
    target_allocation = allocation

    api_url = ITF_TRADING_API_URL + "portfolio/"
    headers = {'Content-Type': 'application/json'}
    data = {
        "api_key": ITF_TRADING_API_KEY,
        "binance": {
            "api_key": binance_account.api_key,
            "secret_key": binance_account.secret_key,
            "type": "market",
            "allocations": allocation,
        }
    }

    if DEBUG:
        data_copy = deepcopy(data)
        del data_copy["binance"]["api_key"]
        del data_copy["binance"]["secret_key"]
        data_copy["binance"]["api_key"] = data_copy["binance"]["secret_key"] = "*****"
        logging.debug(str(data_copy))
        del data_copy

    response = requests.put(api_url, headers=headers, json=data)
    del data

    logging.debug(response.text)

    if response.status_code == 200:
        response_data = response.json()
        while 'retry_after' in response_data:
            try:
                #  CAREFUL: this will strip ANY of the characters from the string!!!
                # proccess_uuid = response_data['portfolio_processing_request'].strip("/api/portfolio_process/")

                proccess_uuid = response_data['portfolio_processing_request'][len("/api/portfolio_process/"):]

                logging.debug(f"sleeping for {response_data['retry_after']}ms...")
                time.sleep(max([int(response_data['retry_after'])/1000/2, 5]))
                api_url = ITF_TRADING_API_URL + "portfolio_process/" + proccess_uuid
                response = requests.post(api_url, headers=headers,
                                         json={"api_key": ITF_TRADING_API_KEY, })
                response_data = response.json()
            except Exception as e:
                response_data = {'error': str(e)}

        if not 'binance' in response_data:
            logging.error("Error processing trades...\n" + json.dumps(response_data))
            return

        from apps.portfolio.models import Allocation
        logging.info(f'Storing allocation {portfolio.target_allocation}, BTC price is {get_BTC_price()}')

        allocation_object = Allocation.objects.create(
            portfolio=portfolio,
            target_allocation=target_allocation,
            realized_allocation=response_data['binance']['allocations'],
            is_realized=True,
            BTC_value=float(response_data['binance']['value']),
            BTC_price=get_BTC_price()
        )
        realized_allocation = response_data['binance']['allocations']
        logging.info(f'>>>> Porfolio successfully rebalanced at {datetime_from_timestamp(int(time.time()))} '
              f'(final target allocation = {allocation}, realized allocation = {realized_allocation}')

        portfolio.rebalanced_at = datetime.now()
        portfolio.save()

        from apps.portfolio.models.allocation_committee import AllocationCommittee

        # save log of which committees contributed in the allocation
        for ticker in committees_used:
            # for horizon in committees_used[ticker]:  # legacy stuff, we don't distinguish horizons anymore
            for committee_vote in committees_used[ticker]:
                committee_object = AllocationCommittee(
                    allocation_id=allocation_object.id,
                    ticker=ticker,
                    committee_id=committee_vote.committee_id,
                    voted_at=committee_vote.timestamp,
                    vote=committee_vote.vote,
                    horizon=0 #horizon,
                )
                committee_object.save()

    else:
        logging.critical(f'!!!!!!! Got response status code {response.status_code}')
        raise TradingAPIException(f'Error calling trading API: {response.text}')
