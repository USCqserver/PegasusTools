import scipy

from scipy.special import betaincinv


def beta_schedule(a, b, sc, lin_pnts, log_pnts):
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

def args_to_sched(*sched_args):
    nargs = len(sched_args)
    if nargs % 2 != 0:
        raise RuntimeError("Expected even number of points for schedule.")


available_schedules = {
    'pl': ("Piecewise Linear", "t0 s0 t1 s1 ..."),
    'pr': ("Pause and ramp", "t1 sp tp tr"),
    'beta': ("Ramped beta schedule (Boundary cancellation protocol)", "a b sq [sc]")
}


def interpret_schedule(tf, *sched_tokens):
    sched_name = sched_tokens[0]
    sched_args = sched_tokens[1:]
    nargs = len(sched_args)
    try:
        if sched_name == "pl":
            pnts = args_to_sched(*sched_args)
            raise NotImplementedError()
        elif sched_name == "pr3":
            if nargs != 4:
                raise RuntimeError("Expected four arguments: t1 sp tp tr")
            t1, sp, tp, tr = float(sched_args[0]), float(sched_args[1]), float(sched_args[2]), float(sched_args[3])
            return ramped_pause_schedule(t1, sp, tp, tr)
        elif sched_name == "beta":
            if nargs != 3:
                raise RuntimeError("Expected at least three arguments: a b sq [sc]")
            a, b, sq = int(sched_args[0]), int(sched_args[1]), float(sched_args[2])
            return ramped_beta_schedule(a, b, sq, tf, 1.0)
        else:
            raise RuntimeError(f"Schedule type {sched_name} not recognized")

    except Exception as e:
        raise RuntimeError(f"An exception occurred interepreting schedule type {sched_name}")
