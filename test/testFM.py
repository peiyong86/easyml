#!/usr/bin/env python
__author__ = 'peiyong'

import sys

sys.path.insert(0, '../')
import unittest

from easyml import FM, FMParam, DataSet


class TestStringMethods(unittest.TestCase):

    def test_fm_AdaDelta_Logloss(self):
        param = FMParam(learning_rate=0.01,
                        embed_size=10,
                        decay=0.8,
                        decay_step=3,
                        epochs=5,
                        batch_size=100,
                        regW=0.0,
                        regV=0.01,
                        loss='log',
                        opt='AdaDelta')
        model = FM(param)
        dataset = DataSet()
        dataset.loaddata('./data/breast_data.txt')
        model.fit(dataset)
        eval_re = model.evaluate(dataset)

        self.assertGreater(eval_re['auc'], 0.9)
        self.assertGreater(eval_re['accuracy'], 0.9)

    def test_fm_SGD_MSELoss(self):
        param = FMParam(learning_rate=0.01,
                        embed_size=10,
                        decay=0.8,
                        decay_step=3,
                        epochs=10,
                        batch_size=100,
                        regW=0.0,
                        regV=0.01,
                        loss='mse',
                        opt='SGD')
        model = FM(param)
        dataset = DataSet()
        dataset.loaddata('./data/breast_data.txt')
        model.fit(dataset)
        eval_re = model.evaluate(dataset)

        self.assertGreater(eval_re['auc'], 0.9)
        self.assertGreater(eval_re['accuracy'], 0.9)

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
