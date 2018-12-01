from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.spatial import distance
import random
import cProfile

def distance_sq(a, b):
    sum = 0
    for i,j in zip(a,b):
        sum += (i-j)**2
    return sum
    #c = a-b
    #c = c*c
    #c = np.sum(c**2)
    #return c

def cost_km(C, U, U_data, u_dict):
    cost = 0
    for i in U:
        c = C.pop()
        C.add(c)
        min_d = distance_sq(U_data[c], U_data[i])
        min_c = c
        for center in C:
            d = distance_sq(U_data[center], U_data[i])
            if d < min_d:
                min_d = d
                min_c = center

        #Update the center in the dictionary:
        cost += min_d
        u_dict[i] = min_c
    return cost


#TODO: Check this code with mattia.
def local_search(U_data, C, U, Z, k, u_dict):
    alpha = 1 #Set it to some high value.
    cost = cost_km(C, U, U_data, u_dict)
    alpha = cost*2
    counter = 0
    while alpha*(1-(0.0001/k)) > cost:
        counter += 1
        C_prime = C.copy()
        u_dict_prime = u_dict.copy()
        alpha = cost

        for i in U:
            min_cost = cost
            temp_u_dict = u_dict.copy()
            if i in C:
                continue
            for c in C.copy(): #Check is c is not changed.
                C.remove(c)
                C.add(i)
                temp_u_dict.clear()
                for c_i in C:
                    temp_u_dict[c_i] = c_i
                for z_i in Z:
                    temp_u_dict[z_i] = -1
                cost_new = cost_km(C, U, U_data, temp_u_dict)
                if (cost_new < min_cost):
                    C_prime = C.copy()
                    min_cost = cost_new
                    u_dict_prime = temp_u_dict.copy()
                C.remove(i)
                C.add(c)
            #C = C_min.copy()
            #u_dict = u_dict_prime.copy()
            cost = min_cost
            if(i%1000 == 0):
                print(str(i)+" done")
        print("Local_search loop %d" % counter)
        C = C_prime.copy()
        u_dict = u_dict_prime.copy()
    return C, u_dict

def process():
    n_rows = 100
    df = pd.read_csv('../shuttle.tst', header=None, sep=' ', nrows=n_rows)
    print(df.shape[0])
    print(df.head())
    print(df.iloc[1:2])

    # Separating out the features
    X = df.loc[:, df.columns != 9].values
    # Separating out the target
    Y = df.loc[:, df.columns == 9].values
    color_dict = {1: 'red', 2: 'blue', 3: 'green', 4: 'black', 5: 'orange', 6: 'brown', 7: 'purple'}
    colors = []
    for i in range(Y.shape[0]):
        val = Y[i][0]
        colors.append(color_dict[val])

    #C = set()
    #C_val = random.sample(range(x.shape[0]), 7)
    #C.update(C_val)
    #print(C)
    #C = {10371, 13927, 169, 12683, 77, 3578, 7386}  # Chosen at random
    C = {34, 38, 43, 13, 81, 56, 58}
    k = 7

    #X = np.array([[1,1],[1,2],[2,1],[2,2],[100,100],[100,101],[101,100],[101,101]])
    #Y = np.array([1,1,1,1,2,2,2,2])
    #C = {0, 1}
    #k = 2



    u_dict = defaultdict(dict)
    for val in C:
        u_dict[val] = val

    U = U = set(i for i in range(X.shape[0]))
    C, u_dict = local_search(X, C, U, set(), k, u_dict)

    #Visualizing the clusters.
    color_dict = defaultdict(str) #Holds the color for each center now.
    colors = ['red', 'blue', 'green', 'black', 'orange', 'brown', 'purple']
    for c, color in zip(C,colors):
        color_dict[c] = color

    colors = []
    for i in range(Y.shape[0]):
        center = u_dict[i]
        colors.append(color_dict[center])

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    x_modified = pca.fit_transform(X)
    #x_modified = tsne.fit_transform(x)
    plt.figure(1)
    plt.scatter(x_modified[:, 0], x_modified[:, 1], c=colors, alpha=0.8)
    plt.savefig('plt.png')
    plt.show()

if __name__ == '__main__':
    #cProfile.run('process()')
    process()
