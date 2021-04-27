import sys
sys.path.append("")
from scipy.io import loadmat
import unittest
from pickle import dump, load
import numpy as np
from GES.GES import GES

class TestGES(unittest.TestCase):
# Learning the causal structure with generalized score-based method and with GES search
# The score function can be negative cross-validated log-likelihood or marginal log-likelihood with regression in RKHS

    # example1
    # for data with single-variate dimensions
    # and score function is negative cross-validated log-likelihood
    def test_single_CV(self):
        example_data1 = load(open("example_data1.pk", 'rb'))
        G_truth = example_data1['G_truth']
        X = example_data1['X']
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
        X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
        maxP = 5 # maximum number of parents when searching the graph
        parameters= {'kfold':10, 'lambda':0.01}
        multi_sign = 0
        score_type = 1
        X = X[:50, :]
        Record = GES(X, score_type, multi_sign, maxP=maxP, parameters=parameters)
        print(Record)


    # example2
    # for data with single-variate dimensions
    # and score function is negative marginal likelihood
    def test_single_marginal(self):
        example_data1 = load(open("example_data1.pk", 'rb'))
        G_truth = example_data1['G_truth']
        X = example_data1['X']
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
        X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
        maxP = 5 # maximum number of parents when searching the graph
        parameters= {'kfold':10, 'lambda':0.01}
        multi_sign = 0
        score_type = 2
        X = X[:50, :]
        Record = GES(X, score_type, multi_sign, maxP=maxP, parameters=parameters)
        print(Record)

    # example3
    # for data with multi-dimensional variables
    # and score function is negative cross-validated log-likelihood
    def test_multi_CV(self):
        example_data = load(open("example_data2.pk", 'rb'))
        Data_save = example_data['Data_save']
        G_save = example_data['G_save']
        d_label_save = example_data['d_label_save']

        trial = 0
        N = G_save[trial].shape[0]
        X = Data_save[trial]
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
        X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
        maxP = 3 # maximum number of parents when searching the graph
        parameters = {'kfold':10,
                      'lambda':0.01,
                      'dlabel':d_label_save[trial]} # indicate which dimensions belong to the i-th variable.
        multi_sign = 1
        score_type = 1
        X = X[:50, :]
        Record = GES(X, score_type, multi_sign, maxP=maxP, parameters=parameters)
        print(Record)

    # example4
    # for data with multi-dimensional variables
    # and score function is negative marginal likelihood
    def test_multi_marginal(self):
        example_data = load(open("example_data2.pk", 'rb'))
        Data_save = example_data['Data_save']
        G_save = example_data['G_save']
        d_label_save = example_data['d_label_save']

        trial = 0
        N = G_save[trial].shape[0]
        X = Data_save[trial]
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
        X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
        maxP = 3  # maximum number of parents when searching the graph
        parameters = {'kfold': 10,
                      'lambda': 0.01,
                      'dlabel': d_label_save[trial]}  # indicate which dimensions belong to the i-th variable.
        multi_sign = 1
        score_type = 2
        X = X[:50, :]
        Record = GES(X, score_type, multi_sign, maxP=maxP, parameters=parameters)
        print(Record)