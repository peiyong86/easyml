#!/usr/bin/env python
import collections
import copy

import numpy as np

from .DataIO import DataGenerator

__author__ = "peiyong"


def majorLabel(labels):
    # 找到数量最多的label并返回
    count = collections.Counter(labels)
    mostkey = max(count.items(), key=lambda x: x[1])
    return mostkey[0]


def singleAttValue(dataset, attindexs):
    """判断属性子集，是否只有一种值"""
    for ind in attindexs:
        if len(dataset.get_att_values(ind)) > 1:
            return False
    return True


def selectBestAtt(dataset, attindexes):
    """first version: information gain"""
    #     current_ent = dataset.entropy()
    entropys = {}
    minAttInd = None
    minEntVal = None
    for ind in attindexes:
        ent = 0
        for val in dataset.get_att_values(ind):
            subset = dataset.get_subset(ind, val)
            weight = len(subset) / len(dataset)
            ent += weight * subset.entropy()
        entropys[ind] = ent

        if minAttInd is None:
            minAttInd = ind
        if minEntVal is None:
            minEntVal = ent
        if ent < minEntVal:
            minEntVal = ent
            minAttInd = ind
    return minAttInd


class TreeNode:
    def __init__(self, label=None, attindex=None, subtrees=None, isleaf=False):
        self.label = label
        self.isleaf = isleaf
        self.attindex = attindex
        if subtrees is None:
            subtrees = {}
        self.subtrees = subtrees


class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, dataset):
        attindex = list(range(dataset.n_features))
        self.root = treeGenerate(dataset, attindex)

    def print_tree(self):
        """Print all nodes in tree"""
        nodes = [self.root]
        while nodes:
            nextnodes = []
            for node in nodes:
                if node.isleaf:
                    print("[Leaf: {}]  ".format(node.label), end="")
                else:
                    print("[Node: att {}]".format(node.attindex), end="")
                nextnodes.extend(node.subtrees.values())
            print("")
            nodes = nextnodes

    def forward(self, x):
        def travel(node, x):
            if node.isleaf:
                return node.label
            ind = node.attindex
            val = x[ind]
            if val in node.subtrees:
                nextnode = node.subtrees[val]
                return travel(nextnode, x)
            else:
                return node.label

        node = self.root
        return travel(node, x)

    def evaluate(self, data):
        labels = []
        preds = []
        for x, y in zip(data.features, data.labels):
            pred = self.forward(x)
            labels.append(y)
            preds.append(pred)

        labels = np.array(labels)
        preds = np.array(preds)

        # auc_score = roc_auc_score(labels, preds)

        correct = 0
        for p, y in zip(preds, labels):
            if p * y > 0:
                correct += 1
        acu_score = correct / len(preds)

        re = {'accuracy': acu_score}
        return re


def treeGenerate(dataset, attindex):
    """生成树算法：
        if D中仅有一种label, or 属性集attindex为空， or attindex对应的属性值仅有1种：
            则返回叶子节点
        否则：
            * 选出最近划分属性
            * 对该属性的每个值对应的子集建树
    """
    labels = dataset.labels
    # 如集合中仅有一种label，返回叶子节点
    if len(set(labels)) == 1:
        return TreeNode(label=labels[0], isleaf=True)

    # 如属性集为空，返回叶子节点
    if len(attindex) == 0:
        return TreeNode(label=majorLabel(labels), isleaf=True)

    # 如属性值全部相同，返回叶子结点
    if singleAttValue(dataset, attindex):
        return TreeNode(label=majorLabel(labels), isleaf=True)

    # 选择最优划分属性
    pickindex = selectBestAtt(dataset, attindex)

    # 根据所选属性，生成子树
    subattindex = copy.deepcopy(attindex)
    subattindex.remove(pickindex)
    valueset = dataset.get_att_values(pickindex)
    subtrees = {}
    for v in valueset:
        subdataset = dataset.get_subset(pickindex, v)
        subtree = treeGenerate(subdataset, subattindex)
        subtrees[v] = subtree
    return TreeNode(label=majorLabel(labels), attindex=pickindex, subtrees=subtrees)
