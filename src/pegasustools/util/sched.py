import scipy

from scipy.special import betaincinv


def beta_schedule(a, b, sc, lin_pnts, log_pnts):
    """
    Returns a scaled incomplete Beta function schedule
        [ (betaincinv(s_i), s_i) ]
    :param a:
    :param b:
    :param sc:
    :param lin_pnts:
    :param log_pnts:
    :return:
    """
    if a==1 and b==1: # Linear schedule case
        return [[0.0, 0.0], [1.0, 1.0]]
    lin_space = sc / (lin_pnts + 1)

    lin_s = [lin_space * i for i in range(lin_pnts + 2)]

    log_s = []
    d = (1.0 - sc) / 2.0
    for i in range(log_pnts):
        log_s.append(1.0 - d)
        d = d / 2.0
    log_s.append(1.0)

    s_pnts = lin_s + log_s
    sched_pnts = [[betaincinv(a, b, s), s] for s in s_pnts]

    return sched_pnts


def ramped_beta_schedule(a, b, sq, tf, tr, sc=0.9, lin_pnts=3, log_pnts=5):
    sched0 = beta_schedule(a, b, sc, lin_pnts, log_pnts)
    sched1 = [[tf*t, sq*s] for (t, s) in sched0]
    sched1.append([tf + tr, 1.0])
    return sched1


def ramped_pause_schedule(t1, sp, tp, tr):
    t2 = t1 + tp
    sched = [
        [0.0, 0.0],
        [t1, sp],
        [t2, sp],
        [t2 + tr, 1.0]
    ]
    return sched


def rbr_schedule(tr, tf, sr, a, b, sq, sc=0.9, lin_pnts=3, log_pnts=4):
    # initial ramp
    l = sr/tr  # ramp rate
    sched = [[0.0, 0.0], [tr, sr]]
    sched_beta = beta_schedule(a, b, sc, lin_pnts, log_pnts)[1:]
    # beta schedule from ramp end to middle
    sched1 = [[tr + tf * t, sr + (sq-sr)*s] for (t, s) in sched_beta]
    sched += sched1
    # ramp again to end
    dsr2 = 1.0 - sq
    tr2 = dsr2 * l
    sched2 = [[tr + tf + tr2, 1.0]]
    sched += sched2
    return sched


def args_to_sched(tf, *sched_args):
    nargs = len(sched_args)
    if nargs % 2 != 0:
        raise RuntimeError("Expected even number of points for schedule.")
    n=nargs//2
    sched = [[0.0, 0.0]]
    for i in range(n):
        sched.append([tf*float(sched_args[2*i]), float(sched_args[2*i+1])])
    if sched[-1][1] < 1.0:
        raise RuntimeError("Expected final anneal point to end at s=1")
    return sched


available_schedules = {
    'pl': ("Piecewise Linear", "tf t0 s0 t1 s1 ..."),
    'pr': ("Pause and ramp", "t1 sp tp tr"),
    'beta': ("Ramped beta schedule (Boundary cancellation protocol)", "a b sq [sc]"),
    'rbr': ("Ramp/Reverse Anneal (with BCP)/Ramp sequence", "tr sr a b sq")
}


def interpret_schedule(tf, *sched_tokens):
    sched_name = sched_tokens[0]
    sched_args = sched_tokens[1:]
    nargs = len(sched_args)
    try:
        if sched_name == "pl":
            return args_to_sched(*sched_args)
        elif sched_name == "pr":
            if nargs != 4:
                raise RuntimeError("Expected four arguments: t1 sp tp tr")
            t1, sp, tp, tr = float(sched_args[0]), float(sched_args[1]), float(sched_args[2]), float(sched_args[3])
            return ramped_pause_schedule(t1, sp, tp, tr)
        elif sched_name == "beta":
            if nargs != 3:
                raise RuntimeError("Expected at least three arguments: a b sq [sc]")
            a, b, sq = int(sched_args[0]), int(sched_args[1]), float(sched_args[2])
            return ramped_beta_schedule(a, b, sq, tf, 1.0)
        elif sched_name == 'rbr':
            if nargs != 5:
                raise RuntimeError("Expected four arguments: tr a b sq")
            tr, sr, a, b, sq = float(sched_args[0]), float(sched_args[1]), int(sched_args[2]), \
                               int(sched_args[3]), float(sched_args[4])
            return rbr_schedule(tr, tf, sr, a, b, sq)

        else:
            raise RuntimeError(f"Schedule type {sched_name} not recognized")

    except Exception as e:
        raise RuntimeError(f"An exception occurred interepreting schedule type {sched_name}")
