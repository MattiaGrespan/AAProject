from k_means import cost_km
from k_means import distance_sq
from k_means import local_search
from gaussiandataset import create_dataset

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from queue import PriorityQueue

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math


#Z: is the set of outliers.
#z: is the number of outliers removed at every iteration.
#C: is the set of centers.
#U: is the universe of set of points with nxd dimensions.
#u_dict: is the dictionary containing association of every point to its center.

def visualize(C, X, u_dict, f_i):
    color_dict = defaultdict(str)  # Holds the color for each center now.
    colors = ['red', 'blue', 'green', 'black', 'orange', 'brown', 'purple']
    for c, color in zip(C, colors):
        color_dict[c] = color

    colors = []
    for i in range(X.shape[0]):
        center = u_dict[i]
        if center == -1:
            colors.append('silver')
        else:
            colors.append(color_dict[center])

    pca = PCA(n_components=2)
    x_modified = pca.fit_transform(X)
    plt.figure(f_i)
    plt.scatter(x_modified[:, 0], x_modified[:, 1], c=colors, alpha=0.8)
    plt.savefig("plt"+str(f_i)+".png")
    #plt.show()

def binary_search(arr, l, r, x):
    mid = 0
    while l < r:
        mid = int((l + r) / 2)
        if arr[mid] == x:
            return arr[mid], mid

        if x < arr[mid]:
            if mid > 0 and x > arr[mid-1]:
                val = 0
                idx = 0
                if x - arr[mid-1] >= arr[mid] - x:
                    val = arr[mid]
                    idx = mid
                else:
                    val = arr[mid-1]
                    idx = mid-1
                return val, idx
            r = mid
        else:
            if mid < len(arr)-1 and x < arr[mid+1]:
                val = 0
                idx = 0
                if x - arr[mid] >= arr[mid+1] - x:
                    val = arr[mid+1]
                    idx = mid+1
                else:
                    val = arr[mid]
                    idx = mid
                return val, idx
            l = mid+1

    return arr[mid], mid

def outliers_kmeansplusplus(C, U, U_data, Z, z, u_dict):
    U_prime = U - Z
    outliers = set()

    distances = np.zeros(len(U_prime)-len(C))
    distances_idx = np.zeros(distances.shape[0], 'int')
    distances_cdf = np.zeros(distances.shape[0])
    counter = 0
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
            if counter == 0:
                distances_cdf[counter] = min_d
            else:
                distances_cdf[counter] = distances_cdf[counter-1] + min_d
            distances[counter] = min_d
            distances_idx[counter] = i
            counter += 1

    while len(outliers) < z and len(distances) >= 1:
        val = random.randint(distances_cdf[0], distances_cdf[-1])
        val, idx = binary_search(distances_cdf, 0, len(distances_cdf)-1, val)
        outlier = distances_idx[idx]
        outliers.add(outlier)
        u_dict[outlier] = -1
        distances = np.delete(distances, idx)

        distances_cdf = np.zeros(distances.shape[0])
        psum = 0
        d_i = 0
        for d in distances:
            psum += d
            distances_cdf[d_i] = psum
            d_i += 1

    return outliers

# This function returns the outliers and puts the key value for outliers to -1.
def outliers_farthest(C, U, U_data, Z, z, u_dict):
    U_prime = U - Z
    q = PriorityQueue(z)
    outliers = set()

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
                val = q.get()
                if val[0] < min_d:
                    q.put((min_d, i))
                else:
                    q.put(val)
            else:
                q.put((min_d, i))

    while not q.empty():
        next_item = q.get()
        outliers.add(next_item[1])
        u_dict[next_item[1]] = -1
        #print(next_item)
    return outliers


def ls_outlier(U_data, C, k, u_dict, z):
    U = set(i for i in range(U_data.shape[0]))
    Z = set()
    Z = outliers_kmeansplusplus(C, U, U_data, Z, z, u_dict)
    alpha = 1
    cost = cost_km(C, U-Z, U_data, u_dict)
    alpha = cost * 2
    visualize(C, U_data, u_dict, 0)

    counter = 0
    # (1-(0.0001/k))
    hyperparam = 0.85
    while alpha*hyperparam > cost:
        counter += 1
        alpha = cost

        # Step 1: Local search without the outliers. TODO: Here the u_dict is modified but doesnt have the outliers anoymore. The size keeps on reducing.
        C, u_dict = local_search(U_data, C, U-Z, Z, k, u_dict)
        C_prime = C.copy()
        Z_prime = Z.copy()
        u_dict_prime = u_dict.copy()

        # Step 2: After this step u_dict_s2 holds.
        min_u_dict_s2 = u_dict.copy()
        outliers = outliers_kmeansplusplus(C, U, U_data, Z, z, min_u_dict_s2)
        Z_new = outliers | Z
        cost_2 = cost_km(C, U-Z_new, U_data, min_u_dict_s2)
        if alpha*hyperparam > cost_2:
            Z_prime = Z_new
            cost = cost_2
            u_dict_prime = min_u_dict_s2.copy()
        else:
            min_u_dict_s2 = u_dict.copy()
        # 2 main outputs. Z_prime and u_dict_s2


        # Step 3: TODO: Working here. Confused when do we swap ? Like do we find one potential minimal swap fro all U's and C's and just swap once ?
        min_u_dict_s3 = u_dict.copy()
        C_backup = C.copy()
        Z_backup = Z.copy()
        u_dict_backup = u_dict.copy()
        for i in U:  # Check with mattiah about doing U-Z.
            C = C_backup.copy()
            Z = Z_backup.copy()
            u_dict = u_dict_backup.copy()
            if i in C:
                continue
            if i in Z:
                Z.remove(i)
            C_min = C.copy()
            min_cost = cost
            u_dict_s3 = u_dict.copy()
            for c in C.copy():  # Check is c is not changed.
                C.remove(c)
                C.add(i)
                u_dict_s3.clear()
                for c_i in C:
                    u_dict_s3[c_i] = c_i
                for z_i in Z:
                    u_dict_s3[z_i] = -1

                outliers = outliers_kmeansplusplus(C, U, U_data, Z, z, u_dict_s3)
                cost_3 = cost_km(C, U-(Z | outliers), U_data, u_dict_s3)
                if (cost_3 < min_cost):
                    C_min = C.copy()
                    min_cost = cost_3
                    min_u_dict_s3 = u_dict_s3.copy()
                    Z_prime = Z | outliers
                    C_prime = C.copy()
                    u_dict_prime = u_dict_s3.copy()
                    print(C)
                    print(outliers)
                    print(Z_prime)
                C.remove(i)
                C.add(c)
            #C = C_min.copy()
            cost = min_cost
            #u_dict = min_u_dict.copy()
            if(i%1000 == 0):
                print(str(i)+" done")
        #There is no need for C_prime. C containts the updated centers.
        print("LS_Outlier loop %d count %d. C1: %d  Cp: %d" % (counter, len(Z_prime), cost, alpha))
        # (1-(0.0001/k))
        val = alpha*(1-(0.0001/k))
        if alpha*hyperparam > cost:
            C = C_prime.copy()
            Z = Z_prime.copy()
            u_dict = u_dict_prime.copy()
            visualize(C, U_data, u_dict, counter+1)
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
    C = {34, 38, 90, 13, 81, 56, 58}
    k = 7
    z = 7

    x = np.array([[1,1],[1,2],[2,1],[2,2],[101,101],[101,102],[102,101], [102,102], [0,1000], [300,300]])
    y = np.array([1,1,1,1,2,2,2,2,3,4])
    C = {0, 4}
    k = 2
    z = 2

    x = create_dataset()
    C = {0, 1500}
    k = 2
    z = 70

    u_dict = defaultdict(dict)
    for val in C:
        u_dict[val] = val

    C, u_dict = ls_outlier(x, C, k, u_dict, z)

    # Visualizing the clusters.
    color_dict = defaultdict(str)  # Holds the color for each center now.
    colors = ['red', 'blue', 'green', 'black', 'orange', 'brown', 'purple']
    for c, color in zip(C, colors):
        color_dict[c] = color

    colors = []
    for i in range(x.shape[0]):
        center = u_dict[i]
        if center == -1:
            colors.append('silver')
        else:
            colors.append(color_dict[center])

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    x_modified = pca.fit_transform(x)
    # x_modified = tsne.fit_transform(x)
    plt.figure(1)
    plt.scatter(x_modified[:, 0], x_modified[:, 1], c=colors, alpha=0.8)
    plt.savefig('plt.png')
    #plt.show()

if __name__ == '__main__':
    process()
