from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from sklearn.manifold import TSNE

def process():
    '''
    n = 5000000  # number of records in file
    s = 10000  # desired sample size
    skip = sorted(random.sample(range(n), n - s))
    df = pd.read_csv('../SUSY.csv', header=None, sep=',', nrows=1000000)
    #df = pd.read_csv('../SUSY.csv', header=None, sep=',', skiprows=skip)
    '''
    '''
    df = pd.read_csv('../Skin_NonSkin.txt', header=None, sep='	')
    print(df.shape[0])
    print(df.head())
    print(df.iloc[1:2])
    '''

    df = pd.read_csv('../bezdekIris.data', header=None, sep=',')
    print(df.shape[0])
    print(df.head())
    print(df.iloc[1:2])

    # Separating out the features
    x = df.loc[:, df.columns != 4].values
    # Separating out the target
    y = df.loc[:, df.columns == 4].values
    color_dict = {1: 'red', 2: 'blue', 3:'green', 4:'black', 5:'orange', 6:'brown', 7:'purple'}

    #For iris dataset: only
    label_dict = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
    ##############

    colors = []
    for i in range(y.shape[0]):
        val = y[i][0]
        colors.append(color_dict[label_dict[val]])

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors, alpha=0.5)
    plt.show()


    #tsne = TSNE(n_components=2)
    #x_tsne = tsne.fit_transform(x)
    #plt.scatter(x_tsne[:,0], x_tsne[:,1], c = colors)
    #plt.show()
    print('Yes')

if __name__ == '__main__':
    from queue import PriorityQueue
    q = PriorityQueue(2)
    q.put(2)
    q.put(3)

    if q.qsize() == q.maxsize:
        val = q.get()
        if val < 1:
            q.put(1)
        else:
            q.put(val)
    while not q.empty():
        next_item = q.get()
        print(next_item)
    #process()

