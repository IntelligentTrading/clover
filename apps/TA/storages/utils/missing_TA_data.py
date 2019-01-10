

def find_TA_storage_data_gaps(ticker, exchange, storage_class_name, key_suffixes, force_fill=False):

    storage_class = [
        storage_class for storage_class in supported_storage_classes
        if storage_class.__name__ == storage_class_name
    ][0]


    for period in storage_class.get_periods_list():

        storage_object = storage_class(ticker=ticker, exchange=exchange, periods_key=period)
        storage_object.get_value()






from apps.TA.indicators.overlap import sma, ema, wma, dema, tema, trima, bbands, ht_trendline, kama, midprice
from apps.TA.indicators.momentum import adx, adxr, apo, aroon, aroonosc, bop, cci, cmo, dx, macd, mom, ppo, \
    roc, rocr, rsi, stoch, stochf, stochrsi, trix, ultosc, willr
from apps.TA.indicators.events import bbands_squeeze_180min


supported_storage_classes = [
    sma.SmaStorage, ema.EmaStorage, wma.WmaStorage, bbands.BbandsStorage,

    adx.AdxStorage, stoch.StochStorage, macd.MacdStorage,

    bbands_squeeze_180min.BbandsSqueeze180MinStorage,

    willr.WillrStorage,  # still the last one
]