from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import pandas as pd

import time

from sklearn.manifold import TSNE

def process():
    df = pd.read_csv('../SUSY.csv', header=None, sep=',', nrows=10000)
    print(df.shape[0])
    print(df.head())
    print(df.iloc[1:2])

    # Separating out the features
    x = df.loc[:, df.columns != 0].values
    # Separating out the target
    y = df.loc[:, df.columns == 0].values
    color_dict = {0: 'red', 1: 'blue'}
    colors = []
    for i in range(y.shape[0]):
        val = y[i][0]
        colors.append(color_dict[val])

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    #pca = PCA(n_components=2)
    #x_pca = pca.fit_transform(x)
    #plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors)
    #plt.show()


    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x)
    plt.scatter(x_tsne[:,0], x_tsne[:,1], c = colors)
    plt.show()
    print('Yes')

if __name__ == '__main__':
    process()

