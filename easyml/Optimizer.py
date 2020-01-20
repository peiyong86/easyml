#!/usr/bin/env python
__author__ = "peiyong"

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate=0.1, decay=0.5, decay_step=20):
        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_step = decay_step

    @abstractmethod
    def update_weights(self, weights):
        pass

    @abstractmethod
    def lr(self):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate, decay, decay_step):
        super().__init__(learning_rate, decay, decay_step)
        self.step = 0

    def lr(self):
    	lr = self.learning_rate * np.power(self.decay, 
                np.floor(self.step/self.decay_step))
    	return lr

    def update_weights(self, weights, gradients, step):
        self.step = step
        for k,v in weights.items():
            weights[k] = weights[k] - self.lr() * gradients[k]
        # weights = weights - self.lr() * gradients
        return weights


class AdaGrad(Optimizer):
    def __init__(self, learning_rate, weights):
        super().__init__(learning_rate)
        self.states = dict()
        for k,v in weights.items():
            self.states[k] = np.zeros(v.shape)
        self.eps = 1e-6

    def lr(self):
        return self.learning_rate

    def update_weights(self, weights, gradients, *args, **args2):
        for k, v in gradients.items():
            self.states[k] += np.power(v, 2)
            weights[k] -= self.learning_rate * v / np.sqrt(self.states[k] + self.eps)
        return weights


class RMSProp(Optimizer):
    """
    Root Mean Square Prop.
    """
    def __init__(self, learning_rate, weights, gamma=0.9):
        super().__init__(learning_rate)
        self.states = dict()
        for k,v in weights.items():
            self.states[k] = np.zeros(v.shape)
        self.eps = 1e-6
        self.gamma = gamma

    def lr(self):
        return self.learning_rate

    def update_weights(self, weights, gradients, *args, **args2):
        for k, v in gradients.items():
            self.states[k] = self.states[k] * self.gamma + (1 - self.gamma) * np.power(v, 2)
            weights[k] -= self.learning_rate * v / np.sqrt(self.states[k] + self.eps)
        return weights


class AdaDelta(Optimizer):
    """
    Similar to RMSProp, AdaDelta compute weighted average of square of gradient value as states.
    Different to RMSProp, AdaDelta doesn't need an initial learning rate.
    """
    def __init__(self, weights, gamma=0.9):
        super().__init__()
        self.states = dict()
        self.deltas = dict()
        for k,v in weights.items():
            self.states[k] = np.zeros(v.shape)
            self.deltas[k] = np.zeros(v.shape)

        self.eps = 1e-6
        self.gamma = gamma

    def lr(self):
        """
        AdaDelta doesn't have a learning rate.
        """
        return None

    def update_weights(self, weights, gradients, *args, **args2):
        for k, v in gradients.items():
            self.states[k] = self.states[k] * self.gamma + (1 - self.gamma) * np.power(v, 2)
            graident_delta = np.sqrt( (self.deltas[k] + self.eps) / (self.states[k] + self.eps) ) * v
            weights[k] -= graident_delta
            self.deltas[k] = self.deltas[k] * self.gamma + (1 - self.gamma) * np.power(graident_delta, 2)
        return weights

