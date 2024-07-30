import numpy as np

from scanpro.gaussq2 import gausq2


def test_gausq2():
    """Test gausq2 function"""
    n = 4
    d = np.array([4.0, 1.0, 3.0, 2.0], dtype=np.float64)
    e = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)  # Note: e(n) is arbitrary
    z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ierr = 0

    d_out, e_out, z_out = gausq2(n, d, e, z, ierr)

    d_soll = [0.26794919, 1.58578644, 3.73205081, 4.41421356]
    e_soll = [6.05845175e-28, 1.65436123e-24, -5.55111512e-17, 0.00000000e+00]
    z_soll = [-0.22985042, 0.14644661, 0.44403692, -0.85355339]

    assert np.allclose(d_soll, d_out)
    assert np.allclose(e_soll, e_out)
    assert np.allclose(z_soll, z_out)
