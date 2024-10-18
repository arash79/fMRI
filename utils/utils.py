from statsmodels.tsa.stattools import acf, pacf
import numpy as np
from scipy import stats


def acf_look_back_period(series, n_lags):
    acf_, confidence_interval = acf(series, nlags=n_lags, alpha=0.05)
    centered_ci = confidence_interval - acf_[:, None]
    outside = np.abs(acf_) >= centered_ci[:, 1]
    return np.argmax(~outside)