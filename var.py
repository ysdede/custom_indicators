from typing import Union

import numpy as np
from jesse.helpers import get_candle_source, slice_candles


def var(candles: np.ndarray, length: int = 2, source_type="close", sequential=False) -> \
        Union[float, np.ndarray]:
    """
    :param candles: np.ndarray
    :param length: int - default: 2
    :param source_type: str - default: close
    :param sequential: bool - default: False
    :return: Union[float, np.ndarray]
    """

    # VIDYA (Chande's Variable Index Dynamic Average) http://www.fxcorporate.com/help/MS/NOTFIFO/i_Vidya.html
    # github.com/ysdede
    # All the VIDYA values are calculated automatically.
    # First of all, the CMO (Chande Momentum Oscillator) value is calculated using the following formula:
    # CMOi = (UpSumi - DnSumi) / (UpSumi + DnSumi)
    #
    # where:
    # UpSumi - is the sum of positive price increments of the current period.
    # DnSumi - is the sum of negative price increments of the current period.
    #
    # This CMO value is then used to calculate the VIDYA indicator:
    #
    # VIDYAi = Pricei x F x ABS(CMOi) + VIDYAi-1 x (1 - F x ABS(CMOi))
    #
    # where:
    # VIDYAi - is the value of the current period.
    # Pricei - is the source price of the period being calculated.
    # F = 2/(Period_EMA+1) - is a smoothing factor.
    # ABS(CMOi) - is the absolute current value of CMO.
    # VIDYAi-1 - is the value of the period immediately preceding the period being calculated.

    if length < 1:
        raise ValueError('Bad parameters.')

    # Accept normal array too.
    if len(candles.shape) == 1:
        source = candles
    else:
        candles = slice_candles(candles, sequential)
        source = get_candle_source(candles, source_type=source_type)

    valpha = 2 / (length + 1)
    change = np.diff(source, prepend=source[0])

    vud1 = np.copy(change)
    vdd1 = np.copy(change)

    vud1 = np.where(vud1 >= 0, vud1, 0)
    vdd1 = np.where(vdd1 >= 0, 0, -vdd1)
    vUD = np.convolve(vud1, np.ones(9, dtype=int), 'valid')
    vDD = np.convolve(vdd1, np.ones(9, dtype=int), 'valid')
    chandeMO = np.abs(np.true_divide(np.subtract(vUD, vDD), np.add(vUD, vDD)))
    chandeMO = np.pad(chandeMO, (source.size - chandeMO.size, 0), 'constant')

    VAR = np.full_like(source, 0.0)
    for i in range(length, VAR.size):
        VAR[i] = (valpha * chandeMO[i] * source[i]) + (1 - valpha * chandeMO[i]) * VAR[i - 1]

    return VAR if sequential else VAR[-1]
