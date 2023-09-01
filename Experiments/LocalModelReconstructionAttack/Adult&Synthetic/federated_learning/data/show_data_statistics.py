import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

def TwoSampleT2Test(X, Y):
    nx, p = X.shape
    ny, _ = Y.shape
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
    t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
    statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))

    f_function = scipy.stats.f(p, nx+ny-p-1)
    p_value = 1 - f_function.cdf(statistic)
    return statistic, p_value


if __name__ == '__main__':
    numbers = []
    with open('all_data.json', 'rb') as f:
        data = json.load(f)

    for user_data in data['user_data'].values():
        print(user_data['cluster_id'])

    for i in range(10):
        with open('train/' + str(i) + ".json", 'rb') as f:
            data = json.load(f)
            print(i, len(data['y']))
        numbers.append(len(data['y']))
    numbers = np.asarray(numbers)
    print(
        f"Sum {np.sum(numbers)} Mean {np.mean(numbers)} std {np.sqrt(np.var(numbers))} Min {np.min(numbers)} Max {np.max(numbers)}")
