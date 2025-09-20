import numpy as np
cimport numpy as cnp
from libc.math cimport isnan

def compute_P(
    cnp.ndarray[cnp.int64_t, ndim=2] Spikes_test,
    cnp.ndarray[cnp.double_t, ndim=2] pext,
    cnp.ndarray[cnp.double_t, ndim=1] pext_A
):
    """
    Spikes_test : (N, T_test) array of 0/1 integers
    pext        : (N, F) array of probabilities
    pext_A      : (F,) array of multipliers
    Returns:
        P : (F, T_test) array
    """
    cdef Py_ssize_t N = Spikes_test.shape[0]
    cdef Py_ssize_t T_test = Spikes_test.shape[1]
    cdef Py_ssize_t F = pext.shape[1]

    cdef cnp.ndarray[cnp.double_t, ndim=2] P = np.empty((F, T_test), dtype=np.float64)
    cdef double[:, ::1] P_mv = P
    cdef long long[:, ::1] spikes = Spikes_test
    cdef double[:, ::1] pext_mv = pext
    cdef double[::1] pextA = pext_A

    cdef Py_ssize_t t, i, f
    cdef double prod

    for t in range(T_test):
        # initialize with 1
        for f in range(F):
            prod = 1.0
            for i in range(N):
                if spikes[i, t] == 1:
                    if not isnan(pext_mv[i, f]):
                        prod *= pext_mv[i, f]
                else:
                    if not isnan(pext_mv[i, f]):
                        prod *= (1.0 - pext_mv[i, f])
            P_mv[f, t] = prod * pextA[f]

    return P