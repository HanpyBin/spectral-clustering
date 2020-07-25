#!/usr/bin/env python
# coding: utf-8

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
from func.tools import getW, getD, getL, get_dis_matrix, getEigen
from io_helper.io_helper import load_data

cluster_num = 3
KNN_k = 10

data = load_data('data\Spiral_cluster=3.txt')
W = getW(data, KNN_k, method='gauss')
D = getD(W)
L = getL(D, W)
eigvec = getEigen(L, cluster_num)
clf = KMeans(n_clusters=cluster_num)
clf.fit(eigvec)
plt.scatter(data[:,0], data[:, 1], c=clf.labels_)