# This file outlines the necessary steps for calculating and obtaining the Equal Error Rate (EER)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import numpy as np

# y_true: array, shape = n_samples. True binary lables {-1,1} - test data?
y= np.array([-1,-1,1,1])
# y_score: array, shape = n_samples. Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
# prediction data?
y_score = np.array([0.1, 0.4, 0.35, 0.8])
# The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise an error will be raised
fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#OR
fnr = 1 - tpr
EER = fpr(np.nanargmin(np.absolute((fnr - fpr))))

N = get_total_negative()
P = get_total_positive()
FP = get_false_positives()
FN = get_false_negatives()

false_positive_rate = FP / N
false_negative_rate = FN / P


def get_total_negative():
    return 0

def get_total_positive():
    return 0

def get_false_positives():
    return 0

def get_false_negatives():
    return 0
