import json
import logging
import time

import requests
import schedule
from django.core.management.base import BaseCommand

from apps.TA.resources.historical_data import load_ticker
from apps.channel.tickers import Tickers, get_usdt_rates_for
from apps.common.utilities.multithreading import start_new_thread
from settings import EXCHANGE_MARKETS, DEBUG

logger = logging.getLogger(__name__)

class Command(BaseCommand):  # fetch_ohlc_tickers
    help = "Fetch tickers every 1 minute"

    def handle(self, *args, **options):
        logger.info(f'>>> Getting ready to fetch ohlc tickers from: {", ".join(EXCHANGE_MARKETS)}')

        usdt_rates = get_usdt_rates_for('BTC', 'ETH')

        #fetch_and_process_all_exchanges(usdt_rates); return # one iteration for debug only

        schedule.every(1).minutes.do(fetch_and_process_binance, usdt_rates)
        if DEBUG:
            fetch_and_process_binance(usdt_rates) # and go now too!

        keep_going = True
        while keep_going:
            try:
                schedule.run_pending()
                time.sleep(5)
            except Exception as e:
                logger.debug(str(e))
                logger.info(">>> Fetching shut down")
                keep_going = False


def fetch_and_process_binance(usdt_rates):
    logger.debug(f'Starting fetch_and_process_binance')
    tickers = Tickers(exchange='binance', usdt_rates=usdt_rates)
    tickers.run()

    send_ohlc_data_to_TA(tickers)
    # send_ohlc_data_to_queue(tickers)

@start_new_thread
def send_ohlc_data_to_TA(tickers_object):
    for symbol, symbol_info in tickers_object.tickers.items():

        if tickers_object._symbol_allowed(symbol_info=symbol_info,
                                          usdt_rates=tickers_object.usdt_rates,
                                          minimum_volume_in_usd=tickers_object.minimum_volume_in_usd):

            ticker = symbol.replace("/","_")
            data = {
                'exchange': tickers_object.exchange,
                'ticker': symbol_info['symbol'],
                'timestamp': int(symbol_info['timestamp'] / 1000),  # milliseconds -> sec
                'open_price': symbol_info['open'],
                'high_price': symbol_info['high'],
                'low_price': symbol_info['low'],
                'close_price': symbol_info['close'],
                'close_volume': symbol_info['baseVolume'],
            }

            try:
                database_response = load_ticker(ticker, data)
                logger.debug(str(database_response))
            except Exception as e:
                logger.debug(str(e))
