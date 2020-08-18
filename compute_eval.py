import cv2
import numpy as np
import sys
from scipy.optimize import minimize
from astropy.stats import sigma_clip


def calcmean3(array1, array2, array3, i, j, k):
    return ((array1 * i) + (array2 * j) + (array3 * k)) / (i+j+k)

def calcmean2(array1, array2, i, j):
    return (((array1 * i) + (array2 * j))/ (i+j))

def compute_errors(gt, pred):
    #print(gt[100], pred[100])
    #print(len(gt), len(pred))
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def compute_rmse(gt, pred):

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse

def compute_d1(gt, pred):

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    return 1.0 - d1

def compute_log10(gt, pred):
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    return log10
