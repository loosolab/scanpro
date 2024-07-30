import numpy as np


def gausq2(n, d, e, z, ierr=0):
    """This function finds the eigenvalues and first components of the eigenvectors
    of a symmetric tridiagonal matrix by the implicit QL method.
    Adapted from Fortran code by Gordon Smyth

    :param int n: The order of the matrix.
    :param np.ndarray d: Contains the diagonal elements of the input matrix.
    :param np.ndarray e: Contains the subdiagonal elements of the input matrix in its first n-1 positions.
    :param np.ndarray z: Contains the subdiagonal elements of the input matrix in its first n-1 positions.
    :return np.ndarray: Eigenvalues in ascending order and the first components of the orthonormal eigenvectors of the symmetric tridiagonal matrix.
    """
    d = np.array(d, dtype=np.float64)
    e = np.array(e, dtype=np.float64)
    z = np.array(z, dtype=np.float64)

    machep = 2.0**(-52.0)

    if n == 1:
        return d, e, z, ierr

    e[n - 1] = 0.0
    for index in range(n):
        j = 0
        while True:
            for m in range(index, n):
                if m == n - 1 or abs(e[m]) <= machep * (abs(d[m]) + abs(d[m + 1])):
                    break
            if m == index:
                break
            if j == 30:
                ierr = index + 1
                return d, e, z, ierr
            j += 1

            g = (d[index + 1] - d[index]) / (2.0 * e[index])
            r = np.sqrt(g * g + 1.0)
            g = d[m] - d[index] + e[index] / (g + np.copysign(r, g))
            s = 1.0
            c = 1.0
            p = 0.0

            for ii in range(m - index):
                i = m - ii - 1
                f = s * e[i]
                b = c * e[i]
                if abs(f) < abs(g):
                    c = g / f
                    r = np.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = np.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s *= c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[index] -= p
            e[index] = g
            e[m] = 0.0

    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]

        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]

        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, e, z
