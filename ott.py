from collections import namedtuple

import custom_indicators as cta
import numba
import numpy as np
from jesse.helpers import get_candle_source, slice_candles

OTT = namedtuple('OTT', ['ott', 'mavg', 'longStop', 'shortStop', 'MT'])


def ott(candles: np.ndarray, length: int = 2, percent: float = 1.4, source_type="close", sequential=False) -> OTT:
    """
    :param candles: np.ndarray
    :param length: int - default: 2
    :param percent: int - default: 1.4
    :param ma_type: str - default: var
    :param source_type: str - default: close
    :param sequential: bool - default: False
    :return: Union[float, np.ndarray]
    """

    # Optimized Trend Tracker https://www.tradingview.com/script/zVhoDQME
    # Author: Anıl Özekşi, Pinescript Developer: KivancOzbilgic
    # Python port: github.com/ysdede
    # This port is limited to vidya (VAR) only. More ma's can indeed be added to make it better.

    if length < 1 or percent <= 0:
        raise ValueError('Bad parameters.')

    # Accept normal array too.
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)

    MAvg = cta.var(source, length, source_type, sequential=True)

    ott_series, longStop, shortStop, MT = ott_fast(MAvg, percent, length)

    if sequential:
        return OTT(ott_series, MAvg, longStop, shortStop, MT)
    else:
        return OTT(ott_series[-1], MAvg[-1], longStop[-1], shortStop[-1], MT[-1])


@numba.njit
def ott_fast(MAvg, percent, length):
    fark = np.multiply(np.multiply(MAvg, percent), 0.01)
    # >>>>>>>
    longStop = np.subtract(MAvg, fark)
    longStopPrev = np.copy(longStop)

    for i in range(length, longStop.size):
        if MAvg[i] > longStop[i - 1]:
            longStop[i] = max(longStop[i], longStop[i - 1])
            longStopPrev[i] = longStop[i - 1]

    for i in range(length, longStop.size):
        if MAvg[i] > longStopPrev[i]:
            longStop[i] = max(longStop[i], longStopPrev[i])
            longStopPrev[i] = longStop[i - 1]
    # <<<<<>>>>>>>
    shortStop = np.add(MAvg, fark)
    shortStopPrev = np.copy(shortStop)
    for i in range(length, shortStop.size):
        if MAvg[i] < shortStop[i - 1]:
            shortStop[i] = min(shortStop[i], shortStop[i - 1])
            shortStopPrev[i] = shortStop[i - 1]

    for i in range(length, shortStop.size):
        if MAvg[i] < shortStopPrev[i]:
            shortStop[i] = min(shortStop[i], shortStopPrev[i])
            shortStopPrev[i] = shortStop[i - 1]
    # <<<<<>>>>>>>
    dir = np.full_like(longStop, 1)
    temp_dir = dir[length - 1]
    for i in range(length, dir.size):
        if temp_dir < 0 and MAvg[i] > shortStopPrev[i]:
            temp_dir = dir[i] = 1
        elif temp_dir < 0 and MAvg[i] < shortStopPrev[i]:
            temp_dir = dir[i] = -1

        elif temp_dir > 0 and MAvg[i] < longStopPrev[i]:
            temp_dir = dir[i] = -1
    # <<<<<>>>>>>>

    MT = np.where(dir > 0, longStop, shortStop)
    OTT = np.where(MAvg > MT, MT * (200 + percent) / 200, MT * (200 - percent) / 200)
    return np.concatenate((np.full(2, 0), OTT[:-2])), longStop, shortStop, MT
