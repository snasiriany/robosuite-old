"""
Numba utils.
"""
import numba

ENABLE_NUMBA = True

def jit_decorator(func):
    if ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=False)(func)
    return func