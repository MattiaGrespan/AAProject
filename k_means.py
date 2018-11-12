from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import random

def distance(a,b):
    c = a-b
    c = c*c
    c = np.sum(c)
    return c

def cost_km(C, U, u_dict):
    cost = 0
    for i in range(U.shape[0]):
        c = C.pop()
        C.add(c)
        min_d = distance(U[c], U[i])
        min_c = c
        for center in C:
            d = distance(U[center], U[i])
            if d < min_d:
                min_d = d
                min_c = center

        #Update the center in the dictionary:
        cost += min_d
        u_dict[i] = min_c
    return cost


#TODO: Check this code with mattia.
def local_search(U, C, k, u_dict):
    alpha = 1 #Set it to some high value.
    cost = cost_km(C, U, u_dict)
    alpha = cost*2
    while alpha*(1-(0.0001/k)) > cost:
        C_prime = C.copy()
        alpha = cost
        for i in range(U.shape[0]):
            C_min = C.copy()
            min_cost = cost
            for c in C.copy(): #Check is c is not changed.
                C.remove(c)
                C.add(i)
                cost_new = cost_km(C, U, u_dict)
                if (cost_new < min_cost):
                    C_min = C.copy()
                    min_cost = cost_new
                C.remove(i)
                C.add(c)
            C = C_min.copy()
            cost = min_cost
        #There is no need for c prime.
    return C


if __name__ == '__main__':
    df = pd.read_csv('../shuttle.tst', header=None, sep=' ')
    print(df.shape[0])
    print(df.head())
    print(df.iloc[1:2])

    # Separating out the features
    x = df.loc[:, df.columns != 9].values
    # Separating out the target
    y = df.loc[:, df.columns == 9].values
    color_dict = {1: 'red', 2: 'blue', 3: 'green', 4: 'black', 5: 'orange', 6: 'brown', 7: 'purple'}
    colors = []
    for i in range(y.shape[0]):
        val = y[i][0]
        colors.append(color_dict[val])

    #C = set()
    #C_val = random.sample(range(x.shape[0]), 7)
    #C.update(C_val)
    #print(C)
    C = {10371, 13927, 169, 12683, 77, 3578, 7386} #Chosen at random
    k=7

    u_dict = defaultdict(dict)
    for val in C:
        u_dict[val] = val

    local_search(x, C, k, u_dict)

    #Visualizing the clusters.
    color_dict = defaultdict(str) #Holds the color for each center now.
    colors = ['red', 'blue', 'green', 'black', 'orange', 'brown', 'purple']
    for c, color in zip(C,colors):
        color_dict[c] = color

    colors = []
    for i in range(y.shape[0]):
        center = u_dict[i]
        colors.append(color_dict[center])

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors, alpha=0.5)
    plt.show()



