import math
from typing import Union

import numpy as np
import talib
from jesse.helpers import get_candle_source, slice_candles


def kamaf(candles: np.ndarray, length: int = 625, source_type="close", sequential=False) -> Union[float, np.ndarray]:
    """
    :param candles: np.ndarray
    :param length: int - default: 625
    :param source_type: str - default: close
    :param sequential: bool - default: False
    :return: Union[float, np.ndarray]
    """

    # github.com/ysdede
    # Fractional Kaufman Adaptive Moving Average
    # https://www.investopedia.com/terms/k/kaufman-moving-average.asp

    if length < 10:
        raise ValueError('Bad parameters.')

    # Accept normal array too.
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)

    length_real = round(length / 10, 1)

    floor = math.floor(length_real)
    ceil = math.ceil(length_real)

    ceil_fraction = round(length_real % 1, 1)
    floor_fraction = round(1 - ceil_fraction, 1)

    print('Len:', length, 'length_real', length_real, 'Floor:', floor, 'Ceil:',
          ceil, 'floor_fraction', floor_fraction, 'ceil_fraction', ceil_fraction)

    floor_kama = talib.KAMA(source, timeperiod=floor)

    if floor == ceil:
        return floor_kama if sequential else floor_kama[-1]

    ceil_kama = talib.KAMA(source, timeperiod=ceil)
    # fractional_ma = np.add(np.multiply(floor_kama, floor_fraction), np.multiply(ceil_kama, ceil_fraction))
    fractional_ma = (floor_kama * floor_fraction) + (ceil_kama * ceil_fraction)
    return fractional_ma if sequential else fractional_ma[-1]
