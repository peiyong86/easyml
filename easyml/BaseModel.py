#!/usr/bin/env python
__author__ = "peiyong"
__date__ = "2021/2/5"

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import roc_auc_score

from .DataIO import DataGenerator
from .Util import Counter, StdLogger
from .Optimizer import SGD, AdaGrad, RMSProp, AdaDelta
from .Loss import LogLoss, MSE, TaylorLoss
from .protobuf import modelweights_pb2


logger = StdLogger()


class BaseAlgo(ABC):
    def __init__(self, param):
        self.param = param
        self.optimizer = None
        self.loss = None
        self.weights = dict()
        self.model_name = None

        self.init_optimizer()
        self.init_loss()

    def init_optimizer(self):
        # optimizer
        param = self.param
        if param.opt == 'SGD':
            self.optimizer = SGD(learning_rate=param.learning_rate,
                                 decay=param.decay,
                                 decay_step=param.decay_step)
        elif param.opt == 'AdaGrad':
            self.optimizer = AdaGrad(learning_rate=param.learning_rate)
        elif param.opt == 'RMSProp':
            self.optimizer = RMSProp(learning_rate=param.learning_rate)
        elif param.opt == 'AdaDelta':
            self.optimizer = AdaDelta()
        else:
            raise Exception("Unknow optimizer type {}".format(param.opt))

    def init_loss(self):
        # Loss
        param = self.param
        if param.loss == 'mse':
            self.loss = MSE()
        elif param.loss == 'log':
            self.loss = LogLoss()
        elif param.loss == 'taylor':
            self.loss = TaylorLoss()
        else:
            raise Exception('Unsupported loss type {}'.format(param.loss))

    @abstractmethod
    def cal_forward(self, features):
        """Input data features and return model forward results.
        """
        pass

    @abstractmethod
    def cal_gradient(self, pred, labels, features):
        """Compute gradients of model weights, and returned as dict
        """
        pass

    @abstractmethod
    def init_model(self, feature_len):
        pass

    def update_weights(self, gradients, step):
        self.weights = self.optimizer.update_weights(self.weights, gradients, step)

    def reg_loss(self):
        reg_loss = 0
        for k, v in self.weights.items():
            reg_loss += np.sum(np.power(v, 2))
        reg_loss *= 0.5
        return reg_loss

    def export_to_pb(self, filename):
        weights_pb = modelweights_pb2.ModelWeights()
        weights_pb.model_name = self.model_name

        for k, v in self.weights.items():
            w = modelweights_pb2.Weights()
            w.value.extend(v.flatten())
            w.shape.extend(v.shape)
            weights_pb.weights[k].CopyFrom(w)

        with open(filename, 'wb') as f:
            f.write(weights_pb.SerializeToString())
            f.close()

    def load_from_pb(self, filename):
        weights_pb = modelweights_pb2.ModelWeights()

        with open(filename, 'rb') as f:
            bytes = f.read()
            weights_pb.ParseFromString(bytes)

        self.model_name = weights_pb.model_name
        for k in weights_pb.weights:
            w_shape = weights_pb.weights[k].shape
            w = np.array(weights_pb.weights[k].value).reshape(w_shape)
            self.weights[k] = w


class TrainingProcedure:
    def __init__(self, param=None):
        # defined in base algorithm class
        self.param = param
        self.optimizer = None
        self.loss = None

    def cal_forward(self, data):
        """Implemented in sub algorithm class"""
        pass

    def cal_gradient(self, pred, y, x):
        """Implemented in sub algorithm class"""
        pass

    def update_weights(self, gradients, step):
        """Implemented in base algorithm class"""
        pass

    def reg_loss(self):
        """Implemented in base algorithm class"""
        pass

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
        data_generator = DataGenerator(features,
                                       labels,
                                       self.param.batch_size)
        batch_num = data_generator.get_batch_num()

        for i in range(self.param.epochs):
            for batch_i, batch_data in enumerate(data_generator.data_generator()):
                batch_x, batch_y = batch_data
                pred = self.cal_forward(batch_x)
                gradients = self.cal_gradient(pred, batch_y, batch_x)

                # Update weights
                self.update_weights(gradients, step=i)

                train_loss = self.loss.calculate_loss(pred, batch_y)
                reg_loss = self.reg_loss()

                train_loss_counter += train_loss
                reg_loss_counter += reg_loss
                log_str = ' '.join(map(str,
                                       ['epoch ', i,
                                        ' batch {} / {} '.format(batch_i, batch_num),
                                        ' lr ', self.optimizer.lr(),
                                        ' loss ', train_loss,
                                        ' reg loss ', reg_loss]))
                # logger.log(log_str)

            re = self.evaluate(data)

            mean_train_loss = train_loss_counter.getvalue()
            mean_reg_loss = reg_loss_counter.getvalue()
            log_str = ' '.join(map(str,
                                   ['epoch ', i,
                                    ' lr ', self.optimizer.lr(),
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
        data_generator = DataGenerator(data.features,
                                       data.labels,
                                       self.param.predict_batch_size,
                                       yield_last=True)
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
        for p, y in zip(preds, labels):
            if p * y > 0:
                correct += 1
        acu_score = correct / len(preds)

        re = {'auc': auc_score, 'accuracy': acu_score}
        return re

    def predict(self, data, verbose=False):
        data_generator = DataGenerator(data.features,
                                       None,
                                       self.param.predict_batch_size,
                                       yield_last=True)
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


