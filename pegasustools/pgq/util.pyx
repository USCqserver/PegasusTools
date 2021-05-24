# distutils: language = c++
# cython: language_level = 3

Pegasus0Shift = [2, 2, 10, 10, 6, 6, 6, 6, 2, 2, 10, 10]


cdef vert2horz(int w, int k, int z):
    """
    Gets the values of w, z, and k of the 4 vertical K_44 counterpart qubits
    to the vertical qubit in (u=0, w, k, z)
    """
    # Evaluate the raw XY coordinates from vertical coordinates
    t = k // 4
    xv = 3 * w + t
    yv = 2 + 3*z + (2 * t) % 3

    # Convert
    z2 = (xv - 1)//3
    w2 = yv // 3
    k02 = (yv % 3) * 4
    return w2, k02, z2


cdef horz2vert(int w, int k, int z):
    """
    Gets values of w and z for the K_44 counterpart qubits
    to the horizontal qubit in (u=1, w, k, z)
    """
    #  Evaluate the raw XY coordinates from horizontal coordinates
    t = k // 4
    xh = 1 + 3*z + (2 * (t + 2)) % 3
    yh = 3 * w + t

    z2 = (yh - 2) // 3
    w2 = xh // 3
    k02 = (xh % 3) * 4
    return w2, k02, z2,


cdef internal_coupling(u, w, k, z, j):
    """
    Gets the internal coupling of opposite parity located at index j
    :param w:
    :param k:
    :param z:
    :return:
    """
    # d1 = 1 if j < Pegasus0Shift[k // 2] else 0
    # d2 = 1 if k < Pegasus0Shift[6 + (j // 2)] else 0
    # return z + d1, j, w - d2
    if u == 0:
        d1 = 1 if j < Pegasus0Shift[k // 2] else 0
        d2 = 1 if k < Pegasus0Shift[6 + (j // 2)] else 0
        return z+d1, j, w-d2
    else:
        d1 = 1 if k < Pegasus0Shift[(j // 2)] else 0
        d2 = 1 if j < Pegasus0Shift[6 + (k // 2)] else 0
        return z+d2, j, w-d1