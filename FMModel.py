#!/usr/bin/env python
__author__ = "peiyong"

import numpy as np
from sklearn.metrics import roc_auc_score

from .DataIO import DataGenerator
from .Loss import LogLoss, MSE, TaylorLoss
from .Util import Counter, StdLogger, clip_gradient


logger = StdLogger()


class FMParam:
    def __init__(self,
                 lr=0.01,
                 embed_size=10,
                 init_stdev=0.1,
                 decay=0.5,
                 decay_step=20,
                 epochs=100,
                 batch_size=128,
                 regW=0,
                 regV=0.01,
                 loss='mse',
                 predict_batch_size=1000):
        self.lr = lr
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


class FM:
    def __init__(self, param=None):
        self.param = param
        if param.loss == 'mse':
            self.loss = MES()
        elif param.loss == 'log':
            self.loss = LogLoss()
        elif param.loss == 'taylor':
            self.loss = TaylorLoss()
        else:
            raise Exception('Unsupported loss type {}'.format(param.loss))

        # weights
        self.w = None
        self.embed = None
        self.b = None

        # train param
        self.lr = param.lr
        self.embed_size = param.embed_size
        self.iter_num = param.epochs
        self.batch_size = param.batch_size
        self.predict_batch_size = param.predict_batch_size

    def init_model(self, feature_len):
        self.w = np.zeros(feature_len)
        self.b = 0
        self.embed = np.random.normal(scale=self.param.init_stdev, 
        	size=(feature_len, self.embed_size))

    def update_weights(self, lr, gradient_w, gradient_embed, gradient_b):
        # self.w = self.w - clip_gradient(lr * gradient_w)
        # self.embed = self.embed - clip_gradient(lr * gradient_embed)
        # self.b = self.b - clip_gradient(lr * gradient_b)
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
        if self.embed_size == 0:
        	cross_part = 0
        forward = linear_part + cross_part + self.b
        return forward

    def cal_gradient(self, pred, labels, features):
        fore_gradient = self.loss.calculate_foregradient(pred, labels)
        # print("fore_gradient {}".format(fore_gradient))
        gradient_w = np.multiply(features, np.expand_dims(fore_gradient, 1))
        gradient_w = np.mean(gradient_w, 0)

        gradient_embed = []
        for i,fea in enumerate(features):
            re = np.multiply(self.embed, np.expand_dims(fea, 1))
            re = np.sum(re, 0)  # sum of x*v
            re2 = np.multiply(np.expand_dims(re, 0), np.expand_dims(fea, 1))
            re3 = np.multiply(np.expand_dims(np.power(fea, 2), 1), self.embed)
            g = re2 - re3
            gradient_embed.append(g*fore_gradient[i])

        gradient_embed = np.array(gradient_embed)
        gradient_embed = np.mean(gradient_embed, 0)

        gradient_b = np.mean(fore_gradient)

        gradient_w += self.param.regW * self.w
        gradient_embed += self.param.regV * self.embed
        # gradient_b += delta * self.b

        # print([gradient_w, gradient_embed, gradient_b])
        return [gradient_w, gradient_embed, gradient_b]

    def reg_loss(self):
        reg_loss = 0.5 * (np.sum(np.power(self.w, 2))
                          + np.sum(np.power(self.embed, 2))
                          + np.power(self.b, 2))
        return reg_loss

    def fit(self, data):
        print("start fit")
        features = data.features
        labels = data.labels
        n_samples, n_features = features.shape
        self.init_model(n_features)

        pre_loss = None
        loss_count = 0

        train_loss_counter = Counter()
        reg_loss_counter = Counter()
        data_generator = DataGenerator(features, labels, self.batch_size)
        batch_num = data_generator.get_batch_num()

        for i in range(self.iter_num):
            lr = self.lr * np.power(self.param.decay,
                                    np.floor(i/self.param.decay_step))

            for batch_i, batch_data in enumerate(data_generator.data_generator()):
                batch_x, batch_y = batch_data
                pred = self.cal_forward(batch_x)
                gradient_w, gradient_embed, gradient_b = self.cal_gradient(pred, batch_y, batch_x)
                self.update_weights(lr, gradient_w, gradient_embed, gradient_b)
                train_loss = self.loss.calculate_loss(pred, batch_y)
                reg_loss = self.reg_loss()

                train_loss_counter += train_loss
                reg_loss_counter += reg_loss
                log_str = ' '.join(map(str,
                                       ['epoch ', i,
                                        ' batch {} / {} '.format(batch_i, batch_num),
                                        ' lr ', lr,
                                        ' loss ', train_loss,
                                        ' reg loss ', reg_loss]))
                # logger.log(log_str)

            re = self.evaluate(data)

            mean_train_loss = train_loss_counter.getvalue()
            mean_reg_loss = reg_loss_counter.getvalue()
            log_str = ' '.join(map(str,
                                   ['epoch ', i,
                                    ' lr ', lr,
                                    ' loss ', mean_train_loss,
                                    ' reg loss ', mean_reg_loss,
                                    ' ', re]))
            logger.log(log_str)

            # step condition
            if not pre_loss:
                pre_loss = mean_train_loss
            else:
                if mean_train_loss > pre_loss:
                    loss_count += 1
                else:
                    loss_count = 0
                pre_loss = mean_train_loss
            # if loss_count == 3:
                # break

    def evaluate(self, data):
        data_generator = DataGenerator(data.features, data.labels, 
        	self.predict_batch_size, yield_last=True)
        labels = []
        preds = []
        for batch_x, batch_y in data_generator.data_generator():
            pred = self.cal_forward(batch_x)
            labels.append(batch_y)
            preds.append(pred)
        
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)

        auc_score = roc_auc_score(labels, preds)

        correct = 0
        for p,y in zip(preds, labels):
            if p*y>0:
                correct += 1
        acu_score = correct/len(preds)

        re = {'auc': auc_score, 'accuracy': acu_score}
        return re


    def predict(self, data, verbose=False):
        data_generator = DataGenerator(data.features, None, self.predict_batch_size, yield_last=True)
        batch_num = data_generator.get_batch_num()

        preds = []
        for batch_i, batch_fea in enumerate(data_generator.data_generator()):
            pred = self.cal_forward(batch_fea)
            preds.append(pred)

            if verbose:
                log_str = ' '.join(map(str,
                                   ['batch {}/{}'.format(batch_i, batch_num)
                                    ]))
                logger.log(log_str)

        preds = np.concatenate(preds)
        return preds



