#!/usr/bin/env python
import collections

import numpy as np
import pandas as pd
from scipy.sparse import *


__author__ = "peiyong"


class Sample:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label


def read_sparse(datafile):
    labels = []

    cols = []
    rows = []
    values = []

    with open(datafile, 'r') as f:
        for i,line in enumerate(f):
            line = line.rstrip().split(' ')
            label = float(line[0])
            label = -1 if label != 1 else 1
            
            col = [int(v.split(":")[0]) for v in line[1:]]
            row = [i]*len(col)
            value = [float(v.split(":")[1]) for v in line[1:]]
            
            labels.append(label)
            rows.extend(row)
            cols.extend(col)
            values.extend(value)
            
    shape = [max(rows)+1, max(cols)+1]

    features = csr_matrix( (values,(rows,cols)), shape=shape )
    labels = np.array(labels)
    return features, labels


def read_dense(datafile):
    """
    each row: [y \t x1, x2, x3 ...]
    """
    labels = []
    features = []
    with open(datafile, 'r') as f:
        for line in f:
            y,x = line.rstrip().split('\t')
            labels.append(float(y))
            features.append(list(map(float, x.split(','))))
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def read_csv(datafile, label_column_name):
    data = pd.pandas.read_csv(datafile)
    # clean up
    data = data.drop(columns=['id'])

    labels = np.array(data[label_column_name])
    data = data.drop(columns=[label_column_name])
    features = np.array(data)

    labels[labels!=1] = -1
    return features, labels


def min_max_normalize(dataset):
    max_val = np.max(dataset.features, axis=0)
    min_val = np.min(dataset.features, axis=0)
    range_val = max_val - min_val
    range_val[range_val==0] = 1
    
    for i in range(len(dataset.features)):
        dataset.features[i] = (dataset.features[i] - min_val) / range_val


class DataSet:
    def __init__(self, features=None, labels=None):
        self.features = features
        self.labels = labels
        self.n_features = None
        self.n_samples = None
        self.sanity_check()

    def loaddata(self, datafile, type='dense'):
        try:
            if type == 'dense':
                self.features, self.labels = read_dense(datafile)
            elif type == 'sparse':
                self.features, self.labels = read_sparse(datafile)
            elif type == 'csv':
                self.features, self.labels = read_csv(datafile, label_column_name = 'y')
            self.sanity_check()
        except Exception as e:
            print(e)
            print("Incorrect input format.")

    def sanity_check(self):
        if str(type(self.features)) == "<class 'scipy.sparse.coo.coo_matrix'>":
            self.features = self.features.tocsr()
        if self.features is not None:
            self.n_features = self.features.shape[1]
            self.n_samples = self.features.shape[0]

    def get_subset(self, attindex, attvalue):
        indexes = self.features[:, attindex] == attvalue
        return DataSet(features=self.features[indexes], labels=self.labels[indexes])

    def entropy(self):
        # labels to probability
        count = collections.Counter(self.labels)
        total = self.n_samples
        probs = [v / total for k, v in count.items()]
        # calculate information entropy
        ent = sum([-p * np.log(p) for p in probs])
        return ent

    def get_att_values(self, pickindex):
        """返回数据集种，属性index为pickindex的值的集合"""
        return set(self.features[:, pickindex])

    def __len__(self):
        return self.n_samples

class DataGenerator:
    def __init__(self, x, y=None, batch_size=None, yield_last=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.data_num = x.shape[0]
        if self.batch_size is None or batch_size > self.data_num:
            self.batch_size = self.data_num
        self.batch_num = int(np.floor(self.data_num/self.batch_size))
        assert self.batch_num > 0
        self.yield_last = yield_last

    @staticmethod
    def sanity_check(data):
        if str(type(data)) == "<class 'scipy.sparse.csr.csr_matrix'>":
            data = data.toarray()
        return data

    def data_generator(self):
        if self.y is None:
            for i in range(self.batch_num):
                batch_x = self.x[i * self.batch_size : (i+1) * self.batch_size]
                yield self.sanity_check(batch_x)
            if self.yield_last and self.data_num > self.batch_num*self.batch_size:
                yield self.sanity_check(self.x[self.batch_size*self.batch_num:])
        else:
            assert self.y.shape[0] == self.x.shape[0]
            for i in range(self.batch_num):
                batch_x = self.x[i * self.batch_size : (i+1) * self.batch_size]
                batch_y = self.y[i * self.batch_size : (i+1) * self.batch_size]
                yield (self.sanity_check(batch_x), batch_y)
            if self.yield_last and self.data_num > self.batch_num*self.batch_size:
                yield (self.sanity_check(self.x[self.batch_size*self.batch_num:],),
                       self.y[self.batch_num*self.batch_size:])

    def get_batch_num(self):
        return self.batch_num
