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
                 epochs=100,
                 batch_size=128,
                 loss='mse'):
        self.lr = lr
        self.embed_size = embed_size
        self.feature_len = feature_len
        self.decay = decay
        self.decay_step = decay_step
        self.epochs = epochs
        self.batch_size=batch_size
        self.loss = loss

class DataGenerator:
    def __init__(self, x, y=None, batch_size=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.data_num = x.shape[0]
        if self.batch_size is None:
            self.batch_size = self.data_num
        self.batch_num = int(np.floor(self.data_num/self.batch_size))
        assert self.batch_num > 0

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
        else:
            assert self.y.shape[0] == self.x.shape[0]
            for i in range(self.batch_num):
                batch_x = self.x[i * self.batch_size : (i+1) * self.batch_size]
                batch_y = self.y[i * self.batch_size : (i+1) * self.batch_size]
                yield (self.sanity_check(batch_x), batch_y)

    def get_batch_num(self):
        return self.batch_num


class Counter:
    def __init__(self, value=0):
        self.value = value
        self.n = 0

    def addvalue(self, value):
        self.value += value
        self.n += 1

    def getvalue(self):
        mean_value = self.value/self.n
        self.resetvalue()
        return mean_value

    def resetvalue(self):
        self.value = 0
        self.n = 0

    def __add__(self, other):
        self.addvalue(other)
        return self


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
        self.iter_num = param.epochs
        self.batch_size = param.batch_size

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

        train_loss_counter = Counter()
        reg_loss_counter = Counter()
        data_generator = DataGenerator(features, labels, self.batch_size)
        batch_num = data_generator.get_batch_num()

        for i in range(self.iter_num):
            lr = 0.01 * np.power(self.param.decay,
                                 np.floor(i/self.param.decay_step))

            for batch_i, batch_data in enumerate(data_generator.data_generator()):
                batch_x, batch_y = batch_data
                pred = self.cal_forward(batch_x)
                gradient_w, gradient_embed, gradient_b = self.cal_gradient(pred, batch_y, batch_x, delta=0.1)
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
                logger.log(log_str)

            mean_train_loss = train_loss_counter.getvalue()
            mean_reg_loss = reg_loss_counter.getvalue()
            log_str = ' '.join(map(str,
                                   ['epoch ', i,
                                    ' lr ', lr,
                                    ' loss ', mean_train_loss,
                                    ' reg loss ', mean_reg_loss]))
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
            if loss_count == 3:
                break

    def predict(self, data):
        data_generator = DataGenerator(data.features, None, self.batch_size)
        batch_num = data_generator.get_batch_num()

        preds = []
        for batch_i, batch_fea in enumerate(data_generator.data_generator()):
            pred = self.cal_forward(batch_fea)
            preds.append(pred)
            log_str = ' '.join(map(str,
                                   ['batch {}/{}'.format(batch_i, batch_num)
                                    ]))
            logger.log(log_str)

        return preds



