#!/usr/bin/env python
__author__ = "peiyong"
__date__ = "2021/2/5"

import numpy as np

from .BaseModel import BaseAlgo, TrainingProcedure



class LRParam:
    def __init__(self,
                 learning_rate=0.01,
                 init_stdev=0.1,
                 decay=0.5,
                 decay_step=20,
                 epochs=100,
                 batch_size=128,
                 regW=0.1,
                 loss='mse',
                 opt ='SGD',
                 predict_batch_size=1000):
        self.learning_rate = learning_rate
        self.init_stdev = init_stdev
        self.decay = decay
        self.decay_step = decay_step
        self.epochs = epochs
        self.batch_size=batch_size
        self.loss = loss
        self.regW = regW
        self.predict_batch_size = predict_batch_size
        self.opt = opt


class LR(BaseAlgo, TrainingProcedure):
    def __init__(self, param=None):
        super().__init__(param)

        # model_name
        self.model_name = 'LogisticRegression'

        # weights
        self.weights = {}

    def init_model(self, feature_len):
        # init weights
        self.weights['w'] = np.zeros(feature_len)
        self.weights['b'] = np.zeros(1)

    def cal_forward(self, features):
        linear_part = np.dot(features, self.weights['w'])
        forward = linear_part + self.weights['b']
        return forward

    def cal_gradient(self, pred, labels, features):
        fore_gradient = self.loss.calculate_foregradient(pred, labels)
        gradient_w = np.multiply(features, np.expand_dims(fore_gradient, 1))
        gradient_w = np.mean(gradient_w, 0)

        gradient_b = np.mean(fore_gradient)
        gradient_w += self.param.regW * self.weights['w']
        # gradient_b += delta * self.weights['b']

        gradients = {'w': gradient_w, 'b': gradient_b}
        return gradients
