#!/usr/bin/env python
__author__ = "peiyong"

import numpy as np

from .BaseModel import BaseAlgo, TrainingProcedure


class FMParam:
    def __init__(self,
                 learning_rate=0.01,
                 embed_size=10,
                 init_stdev=0.1,
                 decay=0.5,
                 decay_step=20,
                 epochs=100,
                 batch_size=128,
                 regW=0,
                 regV=0.01,
                 loss='mse',
                 opt ='SGD',
                 predict_batch_size=1000):
        self.learning_rate = learning_rate
        self.embed_size = embed_size
        self.init_stdev = init_stdev
        self.decay = decay
        self.decay_step = decay_step
        self.epochs = epochs
        self.batch_size=batch_size
        self.loss = loss
        self.regW = regW
        self.regV = regV
        self.predict_batch_size = predict_batch_size
        self.opt = opt


class FM(BaseAlgo, TrainingProcedure):
    def __init__(self, param=None):
        super().__init__(param)

        # model_name
        self.model_name = 'FactorizationMachine'

        # weights
        self.weights = {}

        # train param
        self.embed_size = param.embed_size

    def init_model(self, feature_len):
        # init weights
        self.weights['w'] = np.zeros(feature_len)
        self.weights['b'] = np.zeros(1)
        self.weights['embed'] = np.random.normal(scale=self.param.init_stdev,
                                                 size=(feature_len, self.embed_size))

    def cal_forward(self, features):
        linear_part = np.dot(features, self.weights['w'])
        cross_part = []

        for fea in features:
            # (vx)^2
            re = np.multiply(np.expand_dims(fea, 1), self.weights['embed'])
            re = np.sum(re, 0)
            re = np.power(re, 2)
            # (v^2)*(x^2)
            x_square = np.power(fea, 2)
            v_square = np.power(self.weights['embed'], 2)
            re2 = np.multiply(np.expand_dims(x_square, 1), v_square)
            re2 = np.sum(re2, 0)
            # cross part
            cross_part.append(0.5 * np.sum(re - re2))
        if self.embed_size == 0:
            cross_part = 0
        forward = linear_part + cross_part + self.weights['b']
        return forward

    def cal_gradient(self, pred, labels, features):
        fore_gradient = self.loss.calculate_foregradient(pred, labels)
        gradient_w = np.multiply(features, np.expand_dims(fore_gradient, 1))
        gradient_w = np.mean(gradient_w, 0)

        gradient_embed = []
        for i,fea in enumerate(features):
            re = np.multiply(self.weights['embed'], np.expand_dims(fea, 1))
            re = np.sum(re, 0)  # sum of x*v
            re2 = np.multiply(np.expand_dims(re, 0), np.expand_dims(fea, 1))
            re3 = np.multiply(np.expand_dims(np.power(fea, 2), 1), self.weights['embed'])
            g = re2 - re3
            gradient_embed.append(g*fore_gradient[i])

        gradient_embed = np.array(gradient_embed)
        gradient_embed = np.mean(gradient_embed, 0)

        gradient_b = np.mean(fore_gradient)

        gradient_w += self.param.regW * self.weights['w']
        gradient_embed += self.param.regV * self.weights['embed']
        # gradient_b += delta * self.weights['b']

        gradients = {'w': gradient_w, 'embed': gradient_embed, 'b': gradient_b}
        return gradients
