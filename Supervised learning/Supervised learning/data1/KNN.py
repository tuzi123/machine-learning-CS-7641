
import numpy as np
from scipy import stats
from sklearn import neighbors
from sklearn import cross_validation
import matplotlib.pyplot as plt
import pandas as pd
import timeit

start = timeit.default_timer()
print(start)
learning = np.genfromtxt('DataSet_1/winequality-red.csv', delimiter=';')
learning_in = learning[1:-1,:-1]
learning_out = learning[1:-1, -1]

#learning_in_norm = learning_in
#learning_in_norm[:, 0] = stats.zscore(learning_in[:, 0], ddof=1)
#learning_in_norm[:, 1] = stats.zscore(learning_in[:, 1], ddof=1)
learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)


X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out, test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5, random_state=0)


dist_train_error_k = []
dist_test_error_k = []
uniform_cv_error_k = []

uniform_train_error_k = []
uniform_test_error_k = []
dist_cv_error_k = []
k_neighbors = []

for ind in range(1, 100, 2):
    # make new KNN for uniform voting
    knn_uniform_k = neighbors.KNeighborsClassifier(n_neighbors=ind, weights='uniform')
    # make new KNN with distance based voting
    knn_dist_k = neighbors.KNeighborsClassifier(n_neighbors=ind, weights='distance')
    k_neighbors.append(ind)

    knn_uniform_k.fit(X_train, Y_train)
    # fit distance voting model
    knn_dist_k.fit(X_train, Y_train)

    # scoring for uniform voting
    uniform_train_error_k.append(1-knn_uniform_k.score(X_train, Y_train))
    uniform_cv_error_k.append(1-knn_uniform_k.score(X_CV, Y_CV))
    # scoring for distance based voting
    dist_train_error_k.append(1-knn_dist_k.score(X_train, Y_train))
    dist_cv_error_k.append(1-knn_dist_k.score(X_CV, Y_CV))

fig = plt.figure()
plt.title('KNN Error VS K-Neighbors (uniform)')
plt.xlabel("K-Neighbors")
plt.ylabel("Error")
# plt.ylim(0,.5)
plt.plot(k_neighbors, uniform_train_error_k, 'o-', color="r", label="Training error")
plt.plot(k_neighbors, uniform_cv_error_k, 'o-', color="g", label="Cross Validation error")
plt.legend(loc="best")
fig.savefig('Plots/KNN Error VS K-Neighbors (uniform).png')
plt.close()

fig = plt.figure()
plt.title('KNN Error VS K-Neighbors (distance)')
plt.xlabel("K-Neighbors")
plt.ylabel("Error")
# plt.ylim(0,.5)
plt.plot(k_neighbors, dist_train_error_k, 'o-', color="r", label="Training error")
plt.plot(k_neighbors, dist_cv_error_k, 'o-', color="g", label="Cross Validation error")
plt.legend(loc="best")
fig.savefig('Plots/KNN Error VS K-Neighbors (distance).png')
plt.close()

u_neighbors = k_neighbors[uniform_cv_error_k.index(min(uniform_cv_error_k))]
d_neighbors = k_neighbors[dist_cv_error_k.index(min(dist_cv_error_k))]
print(u_neighbors)
print(d_neighbors)
# make new KNN for uniform voting
knn_uniform = neighbors.KNeighborsClassifier(n_neighbors=u_neighbors, weights='uniform')
# make new KNN with distance based voting
knn_dist = neighbors.KNeighborsClassifier(n_neighbors=d_neighbors , weights='distance')

# scoring for uniform voting
uniform_train_error = []
uniform_test_error = []
uniform_cv_error = []

# scoring for distance based voting
dist_train_error = []
dist_test_error = []
dist_cv_error = []
training_samples = []
for ind in range(100, X_train.shape[0], 100):

    # slice the data
    X_train_step = X_train[0:ind, :]
    Y_train_step = Y_train[0:ind]

    # fit uniform voting model
    knn_uniform.fit(X_train_step, Y_train_step)
    # fit distance voting model
    knn_dist.fit(X_train_step, Y_train_step)

    # scoring for uniform voting
    uniform_train_error.append(1-knn_uniform.score(X_train_step, Y_train_step))
    uniform_cv_error.append(1-knn_uniform.score(X_CV, Y_CV))
    # scoring for distance based voting
    dist_train_error.append(1-knn_dist.score(X_train_step, Y_train_step))
    dist_cv_error.append(1-knn_dist.score(X_CV, Y_CV))

    training_samples.append(ind)

    print(uniform_cv_error[-1])
    print(dist_cv_error[-1])

stop = timeit.default_timer()
print("time to run:", stop-start)

fig = plt.figure()
plt.title('KNN Error VS Training Size (uniform)')
plt.xlabel("Training examples")
plt.ylabel("Error")
# plt.ylim(0,.5)
plt.plot(training_samples, uniform_train_error, 'o-', color="r", label="Training error")
plt.plot(training_samples, uniform_cv_error, 'o-', color="g", label="Cross Validation error")
plt.legend(loc="best")
fig.savefig('Plots/KNN Error VS Training Size (uniform).png')
plt.close()

fig = plt.figure()
plt.title('KNN Error VS Training Size (distance)')
plt.xlabel("Training examples")
plt.ylabel("Error")
# plt.ylim(0,.5)
plt.plot(training_samples, dist_train_error, 'o-', color="r", label="Training error")
plt.plot(training_samples, dist_cv_error, 'o-', color="g", label="Cross Validation error")
plt.legend(loc="best")
fig.savefig('Plots/KNN Error VS Training Size (distance).png')
plt.close()



fig = plt.figure()
plt.title('KNN Learning Curve (distance)')
plt.xlabel("Training examples")
plt.ylabel("Error")
# plt.ylim(0,.5)
plt.plot(training_samples, dist_train_error, '--', color="r", label="Distance Training error")
plt.plot(training_samples, dist_cv_error, 'o-', color="r", label="Distance Cross Validation error")
plt.plot(training_samples, uniform_train_error, '--', color="g", label="Uniform Training error")
plt.plot(training_samples, uniform_cv_error, 'o-', color="g", label="Uniform Cross Validation error")
plt.legend(loc="best")
fig.savefig('Plots/KNN KNN Learning Curve (distance) for both.png')
plt.close()
