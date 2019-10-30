#!/usr/bin/env python
import numpy as np


__author__ = "peiyong"


class Sample:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label


class DataSet:
    def __init__(self, features=None, labels=None):
        self.features = features
        self.labels = labels

        self.sanity_check()

    def loaddata(self, datafile):
        labels = []
        features = []
        with open(datafile, 'r') as f:
            for line in f:
                y,x = line.rstrip().split('\t')
                labels.append(float(y))
                features.append(list(map(float, x.split(','))))
        self.features = np.array(features)
        self.labels = np.array(labels)

    def sanity_check(self):
        if str(type(self.features)) == "<class 'scipy.sparse.coo.coo_matrix'>":
            self.features = self.features.tocsr()


