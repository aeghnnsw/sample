import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_rmse_binned_err(y,y_pred,y_err,fig_path,num_bins=10):
    # plot RMSE with respect to binned error
    # y: true values
    # y_pred: predicted values
    # y_err: error in predicted values
    err_min = np.min(y_err)
    err_max = np.max(y_err)
    err_bins = np.linspace(err_min,err_max,num_bins+1)
    bin_inds = np.digitize(y_err,err_bins)
    bin_inds = bin_inds - 1
    bin_values = (err_bins[:-1]+err_bins[1:])/2
    y = np.array(y)
    y_pred = np.array(y_pred)
    bin_values_corrected = []
    bin_rmse = []
    for i in range(num_bins):
        if len(y[bin_inds==i])>0:
            bin_rmse.append(np.sqrt(np.mean((y[bin_inds==i]-y_pred[bin_inds==i])**2)))
            bin_values_corrected.append(bin_values[i])
    bin_rmse = np.array(bin_rmse)
    bin_values_corrected = np.array(bin_values_corrected)
    fig = plt.figure(figsize=(8,6))
    plt.scatter(bin_values_corrected,bin_rmse)
    # fit bin_rmse and bin_values with no intercept
    a = bin_values_corrected.reshape(-1,1)
    y = bin_rmse.reshape(-1,1)
    m = np.linalg.lstsq(a,y,rcond=None)[0]
    x = np.linspace(err_min,err_max,100)
    mx = m*x
    mx = mx.reshape(-1)
    plt.plot(x,mx,'r--')
    plt.xlabel('Uncertainty bin')
    plt.ylabel('RMSE')
    plt.savefig(fig_path)
    plt.close(fig)

def calc_miscalibration_area(y,y_pred,y_err):
    # calculate miscalibration area
    # and miscalibration at 90% confidence
    # if m_90 is positive, then the model is overconfident
    # if m_90 is negative, then the model is underconfident
    standard_err = np.abs((y-y_pred)/y_err)
    num_samples = len(y)
    qs = np.linspace(1/num_samples,1,num_samples)
    z_scores = norm.ppf(qs)
    c = np.zeros(num_samples)
    for i in range(num_samples):
        c[i] = np.sum(standard_err<=z_scores[i])/num_samples
    ma = np.sum(np.abs(qs-c))/num_samples
    z_90 = norm.ppf(0.90)
    c_90 = np.sum(standard_err<=z_90)/num_samples
    m_90 = c_90/0.90
    return ma,m_90
