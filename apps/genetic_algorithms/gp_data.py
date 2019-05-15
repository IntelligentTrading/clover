import logging

DEBUG = True


# temporarily suspend strategies logging warnings: buy&hold strategy triggers warnings
# as our buy has to be triggered AFTER the minimum strategy initialization period
# determined by the longest_function_history_size parameter of the used grammar
strategy_logger = logging.getLogger("strategies")
strategy_logger.setLevel(logging.ERROR)

TICKS_FOR_PRECOMPUTE = 200



