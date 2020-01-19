#!/usr/bin/env python
__author__ = "peiyong"

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate, decay=0.5, decay_step=20):
        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_step = decay_step

    @abstractmethod
    def update_weights(self, weights):
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
        weights = weights - self.lr() * gradients
        return weights


class AdaGrad(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update_weights(self, weights, gradients):
        weights = weights - self.learning_rate * gradients
        return weights
