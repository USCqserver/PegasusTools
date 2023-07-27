from enum import Enum

import numpy as np
from collections import namedtuple


class EpsilonType(Enum):
    ABSOLUTE = 0
    RELATIVE = 1
    RESIDUAL = 2


def tts(pgs, tf, eps=1.0e-6):
    y=1.0-pgs
    z=np.log(y)
    isnz=(pgs>0.0)
    z=np.where(isnz, z, 1.0)
    ttsarr = tf * np.log(0.01)/z
    ttsarr = np.where(isnz, ttsarr, np.inf)
    return ttsarr

