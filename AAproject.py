#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:08:05 2018

@author: mattiamedinagrespan
"""

import numpy as np

from numpy import genfromtxt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import pandas as pd

import time

from sklearn.manifold import TSNE

#from ggplot import aes, geom_point, ggtitle, ggplot



df=pd.read_csv('SUSY.csv', header = None, sep=',', nrows=10000)

#print(df.shape[0])

#rndperm = np.random.permutation(df.shape[0])

#print(rndperm)




# Separating out the features
x = df.loc[:, df.columns != 0].values

#print(x)
# Separating out the target
y = df.loc[:,df.columns == 0].values
#print(y)

# Standardizing the features
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
# =============================================================================
# df['pca-one'] = principalComponents[:,0]
# df['pca-two'] = principalComponents[:,1] 
# df['pca-three'] = principalComponents[:,2]
# =============================================================================

print (pca.explained_variance_ratio_)


print(principalComponents)


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

#print(principalDf)

finalDf = pd.concat([principalDf,df[[0]]], axis = 1)

print(finalDf)


finalDf.plot(x='principal component 1', y='principal component 2', kind='scatter', style='o')




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [1,0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))



