#!/usr/bin/env python
import numpy as np


__author__ = "peiyong"


class MES:
    def __init__(self):
        pass

    @staticmethod
    def calculate_foregradient(pred, labels):
        fore_gradient = pred - labels
        return fore_gradient
    @staticmethod
    def calculate_loss(pred, labels):
        train_loss = np.mean(0.5 * np.power(pred - labels, 2))
        return train_loss


class StdLogger:
    def __init__(self):
        pass

    def log(self, s):
        print(s)


class FMParam:
    def __init__(self,
                 lr=0.01,
                 embed_size=10,
                 feature_len=10,
                 decay=0.5,
                 decay_step=20,
                 steps=100,
                 loss='mse'):
        self.lr = lr
        self.embed_size = embed_size
        self.feature_len = feature_len
        self.decay = decay
        self.decay_step = decay_step
        self.steps = steps
        self.loss = loss


logger = StdLogger()


class FM:
    def __init__(self, param=None):
        self.param = param
        if param.loss == 'mse':
            self.loss = MES()
        else:
            raise Exception('Unsupported loss type {}'.format(param.loss))

        # weights
        self.w = None
        self.embed = None
        self.b = None

        # train param
        self.lr = param.lr
        self.embed_size = param.embed_size
        self.feature_len = param.feature_len
        self.iter_num = param.steps

    def init_model(self):
        self.w = np.random.rand(self.feature_len)
        self.b = 0
        self.embed = np.random.rand(self.feature_len, self.embed_size)

    def update_weights(self, lr, gradient_w, gradient_embed, gradient_b):
        self.w = self.w - lr * gradient_w
        self.embed = self.embed - lr * gradient_embed
        self.b = self.b - lr * gradient_b

    def cal_forward(self, features):
        linear_part = np.dot(features, self.w)
        cross_part = []

        for fea in features:
            # (vx)^2
            re = np.multiply(np.expand_dims(fea, 1), self.embed)
            re = np.sum(re, 0)
            re = np.power(re, 2)
            # (v^2)*(x^2)
            x_square = np.power(fea, 2)
            v_square = np.power(self.embed, 2)
            re2 = np.multiply(np.expand_dims(x_square, 1), v_square)
            re2 = np.sum(re2, 0)
            # cross part
            cross_part.append(0.5 * np.sum(re - re2))
        forward = linear_part + cross_part + self.b
        return forward

    def cal_gradient(self, pred, labels, features, delta=0.1):
        fore_gradient = self.loss.calculate_foregradient(pred, labels)
        gradient_w = np.multiply(features, np.expand_dims(fore_gradient, 1))
        gradient_w = np.mean(gradient_w, 0)

        gradient_embed = []
        for fea in features:
            re = np.multiply(self.embed, np.expand_dims(fea, 1))
            re = np.sum(re, 0)  # sum of x*v
            re2 = np.multiply(np.expand_dims(re, 0), np.expand_dims(fea, 1))
            re3 = np.multiply(np.expand_dims(np.power(fea, 2), 1), self.embed)
            g = re2 - re3
            gradient_embed.append(g)

        gradient_embed = np.array(gradient_embed)
        gradient_embed = np.mean(gradient_embed, 0)
        gradient_b = np.mean(fore_gradient)
        gradient_w += delta * self.w
        gradient_embed += delta * self.embed
        gradient_b += delta * self.b

        return [gradient_w, gradient_embed, gradient_b]

    def reg_loss(self):
        reg_loss = 0.5 * (np.sum(np.power(self.w, 2))
                          + np.sum(np.power(self.embed, 2))
                          + np.power(self.b, 2))
        return reg_loss

    def fit(self, data):
        self.init_model()
        features = data.features
        labels = data.labels

        pre_loss = None
        loss_count = 0

        for i in range(self.iter_num):
            lr = 0.01 * np.power(self.param.decay,
                                 np.floor(i/self.param.decay_step))
            pred = self.cal_forward(features)
            gradient_w, gradient_embed, gradient_b = self.cal_gradient(pred, labels, features, delta=0.1)
            self.update_weights(lr, gradient_w, gradient_embed, gradient_b)
            train_loss = self.loss.calculate_loss(pred, labels)
            reg_loss = self.reg_loss()
            log_str = ' '.join(map(str, ['iter ', i, ' lr ', lr, ' loss ', train_loss, ' reg loss ', reg_loss]))
            logger.log(log_str)

            # step condition
            if not pre_loss:
                pre_loss = train_loss
            else:
                if train_loss > pre_loss:
                    loss_count += 1
                else:
                    loss_count = 0
                pre_loss = train_loss
            if loss_count == 3:
                break



