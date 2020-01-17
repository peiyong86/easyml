#!/usr/bin/env python
__author__ = "peiyong"

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate_foregradient(self, pred, labels):
        pass

    @abstractmethod
    def calculate_loss(self, pred, labels):
        pass


class MSE(Loss):
    def calculate_foregradient(self, pred, labels):
        fore_gradient = pred - labels
        return fore_gradient

    def calculate_loss(self, pred, labels):
        train_loss = np.mean(0.5 * np.power(pred - labels, 2))
        return train_loss


class LogLoss(Loss):
    def calculate_foregradient(self, pred, labels):
        fore_gradient = -labels*(1.0-1.0/(1.0+np.exp(-labels*pred)))
        return fore_gradient

    def calculate_loss(self, pred, labels):
        train_loss = np.log(np.exp(-pred * labels) + 1.0) 
        return np.mean(train_loss)


class TaylorLoss(Loss):
    def calculate_foregradient(self, pred, labels):
        fore_gradient = 0.25*pred - 0.5*labels
        fore_gradient = np.maximum(fore_gradient, -1)
        fore_gradient = np.minimum(fore_gradient, 1)
        return fore_gradient

    def calculate_loss(self, pred, labels):
        train_loss = np.log(2) - 0.5 * pred * labels + 0.125 * pred * pred
        return np.mean(train_loss)