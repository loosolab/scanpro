import numpy as np
import scipy
from scipy.stats import f, chi2, rankdata
from scipy.special import digamma, polygamma
from statsmodels.nonparametric.smoothers_lowess import lowess
from scanpro.utils import pmin, pmax, gauss_quad_prob


def linkfun(x):
    return x / (1 + x)


def linkinv(x):
    return x / (1 - x)


def fit_f_dist_robust(x, df1, covariate=None, winsor_tail_p=[0.05, 0.1]):
    """Estimate parameters of a scaled F distribution robustly, given first
    degrees of freedom. Robust estimation limit the effect of outliers.
    Method from Gordon Smyth and Belinda Phipson, and
    adapted from the R limma package implementation.

    :param np.ndarray x: List of variances from linear model fitting
    :param np.ndarray df1: First degrees of freedom
    :param np.ndarray covariate: defaults to None
    :param list winsor_tail_p: defaults to [0.05, 0.1]
    :return dict: Scaled prior variances and shrunk prior degrees of freedom
    """

    # initialize variables
    if not isinstance(winsor_tail_p, np.ndarray):
        winsor_tail_p = np.array(winsor_tail_p)

    # initialise results dictionary
    res = {}
    scale = np.nan
    df2 = np.nan
    df2_shrunk = np.nan
    res['scale'] = scale
    res['df2'] = df2
    res['df2_shrunk'] = df2_shrunk

    # check x
    n = len(x)
    if n < 2:
        return res
    if n == 2:
        return (fit_f_dist(x, df1=df1))

    # check df1
    if len(df1) != n:
        print("length of x and df1 are different!")
        return None

    # handle zeros and missing values
    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    notallok = ~ok.all()
    if notallok:
        df2_shrunk = x
        x = x[ok]
        if len(df1 > 1):
            df1 = df1[ok]
        if covariate:
            covariate = covariate[ok]

        # fit F-distribution to corrected data
        fit = fit_f_dist_robust(x, df1, covariate=covariate, winsor_tail_p=winsor_tail_p)
        df2_shrunk[ok] = fit['df2_shrunk']
        df2_shrunk[~ok] = fit['df2']
        if not covariate:
            scale = fit['scale']

        # save results
        res['scale'] = scale
        res['df2'] = fit['df2']
        res['df2_shrunk'] = df2_shrunk

        return res

    # avoid zeros and negativ values
    m = np.median(x)
    if m <= 0:
        print("Variances are mostly <= 0")
        return None
    i = (x < m * 1e-12)
    if i.any():
        n_zero = sum(i)
        if n_zero == 1:
            print("One very small variance detected, has been offset away from zero")
        else:
            print(n_zero + " very small variances detected! have been offset away from zero")
        x[i] = m * 1e-12

    # store non robust results
    non_robust = fit_f_dist(x, df1)

    prob = np.resize(winsor_tail_p, 2)
    winsor_tail_p = np.resize(winsor_tail_p, 2)
    prob[1] = 1 - winsor_tail_p[1]
    if np.all(winsor_tail_p < 1 / n):
        non_robust['df2_shrunk'] = np.resize(non_robust['df2'], n)
        return non_robust

    # transfrom x to constant df1
    if len(df1) > 1:
        df1_max = max(df1)
        i = (df1 < (df1_max - 1e-14))
        if i.any():
            if not covariate:
                s = non_robust['scale']
            else:
                s = non_robust['scale'][i]
            f1 = x[i] / s
            df2 = non_robust['df2']
            p_upper = 1 - f.logcdf(f1, df1[i], df2)  # 1-logcdf to calculate upper tail
            p_lower = f.logcdf(f1, df1[i], df2)
            up = p_upper < p_lower
            if up.any():
                f1[up] = np.log(1 - f.ppf(p_upper[up], df1_max, df2))  # upper tail
            if ~up.any():
                f1[~up] = np.log(f.ppf(p_lower[~up], df1_max, df2))
            x[i] = f1 * s
            df1 = df1_max
        else:
            df1 = df1[0]
    z = np.log(x)
    if not covariate:
        z_trend = scipy.stats.trim_mean(z, winsor_tail_p[1])
        z_resid = z - z_trend
    else:
        # this was not tested, because covariate is at this point None, this should be skipped
        lo = lowess(z, covariate, frac=0.4)
        z_trend = lo[:, 1]
        z_resid = z - z_trend

    # Moments of Winsorized residuals
    zrq = np.quantile(z_resid, q=prob)
    zwins = pmin(pmax(z_resid, zrq[0]), zrq[1])
    zwmean = np.mean(zwins)
    zwvar = np.mean((zwins - zwmean)**2) * n / (n - 1)

    # Theoretical Winsorized moments
    g = gauss_quad_prob(128, dist='uniform')  # g[0] -> nodes, g[1] -> weights

    # Try df2=1e10 instead of df2=np.inf, since np.inf returns nan
    mom = winsorized_moments(df1, 1e10, winsor_tail_p, linkfun, linkinv, g)  # mom[0]=mean, mom[1]=var
    funval_inf = np.log(zwvar / mom[1])
    if funval_inf <= 0:
        df2 = np.inf
        # Correct trend for bias
        z_trend_corrected = z_trend + zwmean - mom[0]
        s20 = np.exp(z_trend_corrected)
        # Posterior df for outliers
        F_stat = np.exp(z - z_trend_corrected)
        tail_p = 1 - chi2.cdf(F_stat * df1, df1)
        r = rankdata(F_stat)
        empirical_tail_prop = (n - r + 0.5) / n
        prop_not_outlier = pmin(tail_p / empirical_tail_prop, 1)
        df_pooled = n * df1
        df2_shrunk = np.resize(df2, n)
        Ou = prop_not_outlier < 1
        if Ou.any():
            df2_shrunk[Ou] = prop_not_outlier[Ou] * df_pooled
            o = np.argsort(tail_p)  # order
            df2_shrunk[o] = np.maximum.accumulate(df2_shrunk[o])

        # add results to dictionary
        res['scale'] = s20
        res['df2'] = df2
        res['df2_shrunk'] = df2_shrunk
        res['tail_p_value'] = tail_p
        return res

    # Use non-robust estimate as lower bound for df2
    if non_robust['df2'] == np.inf:
        non_robust['df2_shrunk'] = np.resize(non_robust['df2'], n)
        return non_robust

    rbx = linkfun(non_robust['df2'])
    funval_low = fun(rbx, df1, linkinv, winsorized_moments, zwvar, winsor_tail_p, linkfun, g)
    if funval_low >= 0:
        df2 = non_robust['df2']
    else:
        # interval [rbx, 0.99] since 1 gives divisionbyzero error
        u = scipy.optimize.brentq(fun, rbx, 0.99, (df1, linkinv, winsorized_moments, zwvar, winsor_tail_p, linkfun, g))
        df2 = linkinv(u)

    # Correct ztrend for bias
    mom = winsorized_moments(df1, df2, winsor_tail_p, linkfun, linkinv, g)
    z_trend_corrected = z_trend + zwmean - mom[0]
    s20 = np.exp(z_trend_corrected)

    # Posterior df for outliers
    z_resid = z - z_trend_corrected
    F_stat = np.exp(z_resid)
    log_tail_p = 1 - f.logcdf(F_stat, df1, df2)
    tail_p = np.exp(log_tail_p)
    r = rankdata(F_stat)
    log_empirical_tail_prop = np.log(n - r + 0.5) - np.log(n)
    log_prop_not_outlier = pmin(log_tail_p - log_empirical_tail_prop, 0)
    prop_not_outlier = np.exp(log_prop_not_outlier)
    prop_outlier = np.expm1(log_prop_not_outlier)

    if np.any(log_prop_not_outlier < 0):
        o = np.argsort(log_tail_p)

        # Find df2_outlier to make max(F_stat) the median of the distribution
        # Exploit fact that log(tail_p) is nearly linearly with positive 2nd deriv as a function of df2
        # Note that min_tail_p and new_tail_p are always less than 0.5
        min_log_tail_p = min(log_tail_p)
        if min_log_tail_p == -np.inf:
            df2_outlier = 0
            df2_shrunk = prop_not_outlier * df2
        else:
            df2_outlier = np.log(0.5) / min_log_tail_p * df2
            # Iterate for accuracy
            new_log_tail_p = 1 - f.logcdf(max(F_stat), df1, df2)  # equivilant to lower.tail = FALSE in R
            df2_outlier = np.log(0.5) / new_log_tail_p * df2_outlier
            df2_shrunk = prop_not_outlier * df2 + prop_outlier * df2_outlier

        # Force df2_shrunk to be monotonic in tail_p
        o = np.argsort(log_tail_p)
        df2_ordered = df2_shrunk[o]
        m = np.maximum.accumulate(df2_ordered)
        m = m / np.array(range(1, n + 1))
        i_min = np.argmin(m)
        df2_ordered[:i_min] = m[i_min]
        df2_shrunk[o] = np.maximum.accumulate(df2_ordered)

    else:
        df2_outlier = df2
        df2_shrunk = np.resize(df2, n)

    # append results to dictionary
    res['scale'] = s20
    res['df2'] = df2
    res['tail_p_value'] = tail_p
    res['prop_outlier'] = prop_outlier
    res['df2_outlier'] = df2_outlier
    res['df2_shrunk'] = df2_shrunk

    return res


def fit_f_dist(x, df1, covariate=None):
    """Moment estimation of the parameters of a scaled F-distribution.
    Method from Gordon Smyth and Belinda Phipson, adapted from
    R limma package implementation.

    :param np.ndarray x: List of variances from linear model fitting.
    :param np.ndarray df1: First degrees of freedom.
    :param np.ndarray covariate: defaults to None.
    :return dict: scaled prior variances and schrunk prior degrees of freedom.
    """

    # initialise results dictionary
    res = {}
    scale = np.nan
    df2 = np.nan
    res['scale'] = scale
    res['df2'] = df2

    # check x
    n = len(x)
    if n == 0:
        return res

    if n == 1:
        res['scale'] = x
        res['df2'] = 0
        return res

    # check df1
    ok = np.isfinite(df1) & (df1 > 1e-15)
    if len(df1) == 1:
        if not ok:
            return res
        else:
            if len(df1) != n:
                print("x and df1 have different lengths")
                return None

    # remove missing, infinite or negative values
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = sum(ok)
    if nok == 1:
        res['scale'] = x[ok]
        res['df2'] = 0
        return res
    notallok = nok < n
    if notallok:
        x = x[ok]
        if len(df1) > 1:
            df1 = df1[ok]

    x = pmax(x, 0)
    m = np.median(x)
    if m == 0:
        print("Warning! More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1
    else:
        if np.any(x == 0):
            print("Warning! Zero sample variances detected, have been offset away from zero.")

    x = pmax(x, 1e-5 * m)
    z = np.log(x)
    e = z - digamma(df1 / 2) + np.log(df1 / 2)
    if not covariate:
        emean = np.mean(e)
        evar = sum(((e - emean)**2) / (nok - 1))
    # estimate scale and df2
    evar = evar - np.mean(polygamma(1, df1 / 2))  # trigamma function
    # calcualte scale and df2
    if evar > 0:
        df2 = 2 * trigamma_inverse(evar)
        scale = np.exp(emean + digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        if not covariate:
            scale = np.mean(x)
        else:
            scale = np.exp(emean)

    res['scale'] = scale
    res['df2'] = df2

    return res


def trigamma_inverse(x):
    """Compute the inverse of trigamma function.

    :param numpy.ndarray x: Numeric vector.
    :return numpy.ndarray: Numeric vector y satisfying trigamma(y)==x.
    """
    # check if x has the right format
    if not isinstance(x, np.ndarray):
        if isinstance(x, list):
            x = np.array(x)
        else:
            x = np.array([x])
    if len(x) == 0:
        return 0

    # Treat out-of-range values as special cases
    omit = np.isnan(x)
    if any(omit):
        y = x
        if any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    omit = x < 0
    if any(omit):
        y = x
        y[omit] = np.nan
        if any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    omit = x > 1e7
    if any(omit):
        y = x
        y[omit] = 1 / np.sqrt(x[omit])
        if (any(~omit)):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    omit = x < 1e-6
    if any(omit):
        y = x
        y[omit] = 1 / x[omit]
        if any(~omit):
            y[~omit] = trigamma_inverse(x[~omit])
        return y

    # Newton's method
    # 1/trigamma(y) is convex, nearly linear and strictly > y-0.5,
    # so iteration to solve 1/x = 1/trigamma is monotonically convergent
    y = 0.5 + 1 / x
    iter = 0
    while iter < 50:
        tri = polygamma(1, y)  # trigamma
        dif = tri * (1 - tri / x) / polygamma(2, y)
        y = y + dif
        if np.max(-dif / y) < 1e-8:
            break
        iter = iter + 1
    return y


def fun(x, df1, linkinv, winsorized_moments, zwvar, winsor_tail_p, linkfun, g):
    """Estimate df2 by matching variance of winsorized residuals.
    """
    df2 = linkinv(x)
    mom = winsorized_moments(df1, df2, winsor_tail_p, linkfun, linkinv, g)
    return np.log(zwvar / mom[1])


def winsorized_moments(df1, df2, winsor_tail_p, linkfun, linkinv, g):
    """Method of winsorized moments.
    Adapted from the R package 'Limma'.

    :return np.ndarray: 2D list, winsorized means and variances
    """
    fq = f.ppf([winsor_tail_p[0], 1 - winsor_tail_p[1]], df1, df2)
    zq = np.log(fq)
    q = linkfun(fq)
    nodes = q[0] + (q[1] - q[0]) * g[0]
    f_nodes = linkinv(nodes)
    z_nodes = np.log(f_nodes)
    f1 = f.pdf(f_nodes, dfn=df1, dfd=df2) / (1 - nodes)**2
    q21 = q[1] - q[0]
    m = q21 * sum(g[1] * f1 * z_nodes) + sum(zq * winsor_tail_p)
    v = q21 * sum(g[1] * f1 * (z_nodes - m)**2) + sum((zq - m)**2 * winsor_tail_p)

    return np.array([m, v])
