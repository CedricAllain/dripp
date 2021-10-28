"""Utils functions used all accross the package"""

import cProfile


def profile_this(fn):
    def profiled_fn(*args, **kwargs):
        filename = fn.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(fn, *args, **kwargs)
        prof.dump_stats(filename)
        return ret
    return profiled_fn
