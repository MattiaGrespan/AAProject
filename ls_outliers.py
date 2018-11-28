from k_means import cost_km
from k_means import distance_sq
from k_means import local_search

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from queue import PriorityQueue

import pandas as pd
import matplotlib.pyplot as plt


#Z: is the set of outliers.
#z: is the number of outliers removed at every iteration.
#C: is the set of centers.
#U: is the universe of set of points with nxd dimensions.
#u_dict: is the dictionary containing association of every point to its center.

#TODO: check this function.
def outliers_farthest(C, U_data, Z, z):
    U = set(i for i in range(U_data.shape(0)))
    U_prime = U - Z
    q = PriorityQueue(z)

    for i in U_prime:
        c = C.pop()
        C.add(c)
        min_d = distance_sq(U_data[c], U_data[i])
        min_c = c
        if i not in C:
            for center in C:
                d = distance_sq(U_data[i], U_data[center])
                if d < min_d:
                    min_d = d
                    min_c = center
            if q.qsize() == q.maxsize:
                val = q.pop()
                if val[0] < min_d:
                    q.put((min_d, i))
                else:
                    q.put(val)
            else:
                q.put((val, i))

    while not q.empty():
        next_item = q.get()
        Z.add(next_item[1])
        #print(next_item)
    return Z


def ls_outlier(U_data, C, k, u_dict, z):
    Z = set()
    Z = outliers_farthest(C, U_data, Z, z)

    alpha = 1
    cost = cost_km(C, U_data, u_dict)
    alpha = cost * 2

    while alpha*(1-(0.0001/k)) > cost:
        alpha = cost

        #Step 1: TODO: Start working here.
        #C = local_search(U_data, C, k, u_dict)

        for i in range(U_data.shape[0]):
            C_min = C.copy()
            min_cost = cost
            min_u_dict = u_dict.copy()
            if i in C:
                continue
            for c in C.copy(): #Check is c is not changed.
                C.remove(c)
                C.add(i)
                u_dict.clear()
                for c_i in C:
                    u_dict[c_i] = c_i
                cost_new = cost_km(C, U_data, u_dict)
                if (cost_new < min_cost):
                    C_min = C.copy()
                    min_cost = cost_new
                    min_u_dict = u_dict.copy()
                C.remove(i)
                C.add(c)
            C = C_min.copy()
            cost = min_cost
            u_dict = min_u_dict.copy()
            if(i%1000 == 0):
                print(str(i)+" done")
        #There is no need for c prime.
    return C, u_dict


def process():
    n_rows = 100
    df = pd.read_csv('../shuttle.tst', header=None, sep=' ', nrows=n_rows)
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

    # C = set()
    # C_val = random.sample(range(x.shape[0]), 7)
    # C.update(C_val)
    # print(C)
    # C = {10371, 13927, 169, 12683, 77, 3578, 7386}  # Chosen at random
    C = {34, 38, 43, 13, 81, 56, 58}
    k = 7

    # x = np.array([[1,1],[1,2],[2,1],[2,2],[100,100],[100,101],[101,100],[101,101]])
    # y = np.array([1,1,1,1,2,2,2,2])
    # C = {0, 1}
    # k = 2

    u_dict = defaultdict(dict)
    for val in C:
        u_dict[val] = val

    C, u_dict = local_search(x, C, k, u_dict)

    # Visualizing the clusters.
    color_dict = defaultdict(str)  # Holds the color for each center now.
    colors = ['red', 'blue', 'green', 'black', 'orange', 'brown', 'purple']
    for c, color in zip(C, colors):
        color_dict[c] = color

    colors = []
    for i in range(y.shape[0]):
        center = u_dict[i]
        colors.append(color_dict[center])

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    x_modified = pca.fit_transform(x)
    # x_modified = tsne.fit_transform(x)
    plt.figure(1)
    plt.scatter(x_modified[:, 0], x_modified[:, 1], c=colors, alpha=0.8)
    plt.savefig('plt.png')
    plt.show()

if __name__ == '__main__':
    process()
