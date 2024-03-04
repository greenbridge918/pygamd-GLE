import numpy as np
from scipy import interpolate

def get_parameters(filename, n_points, rcut):
    x, y = np.loadtxt(filename, delimiter=' ',unpack=True, comments='<')
    interval = np.round(np.diff(x), 5)
    assert np.unique(interval).shape[0] == 1, "All spaces must be equal!!"
    space = np.diff(x)[0]
    assert np.round((x.shape[0]-1) * space, 5) == np.round(rcut, 5), "n_points != rcut/binsize"
    func = interpolate.CubicSpline(x, y)
    _cp = func.c.T.flatten()
    _cf = (func.derivative(1).c.T).flatten()
    paras = np.concatenate((np.asarray([rcut]), np.asarray([n_points]), _cp, _cf, x), axis=0)
    return paras