import numpy as np
import matplotlib.pyplot as plt


def normal_distribute(mu, sigma, count):
    x = np.random.normal(mu, sigma, count)
    x=x.reshape(-1, 1)
    print(x.shape)
    y = np.random.normal(mu, sigma, count)
    y=y.reshape(-1, 1)
    x=np.round(x)
    y=np.round(y)
    return x, y

def create_dataset():
    x1, y1 = normal_distribute(200, 100, 1000)
    x2, y2 = normal_distribute(2000, 100, 1000)
    x3, y3 = normal_distribute(1000, 100, 100)
    x4, y4 = normal_distribute(1000, 1000, 100)

    X = np.vstack((x1,x2))
    X = np.vstack((X,x3))
    X = np.vstack((X, x4))

    Y = np.vstack((y1,y2))
    Y = np.vstack((Y,y3))
    Y = np.vstack((Y, y4))

    #plt.scatter(X,Y)
    #plt.show()
    return np.hstack((X, Y))

if __name__ == '__main__':
    create_dataset()








