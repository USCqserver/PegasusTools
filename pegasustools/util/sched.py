import scipy

from scipy.special import betainc, betaincinv


def beta_schedule(sc, lin_pnts, log_pnts, a, b):
    lin_space = sc / (lin_pnts + 1)

    lin_s = [lin_space * i for i in range(lin_pnts + 1)]

    log_s = []
    d = (1.0 - sc) / 2.0
    for i in range(log_pnts):
        log_s.append(1.0 - d)
        d = d / 2.0
    log_s.append(1.0)

    s_pnts = lin_s + log_s
    sched_pnts = [(betaincinv(a, b, s), s) for s in s_pnts]

    return sched_pnts
