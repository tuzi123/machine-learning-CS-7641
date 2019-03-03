
import numpy as np
from sklearn import svm
from sklearn import cross_validation
import matplotlib.pyplot as plt
from scipy import stats
import timeit
from sklearn.learning_curve import learning_curve
import pandas as pd

start = timeit.default_timer()
print(start)

learning = np.genfromtxt('DataSet_2/spambase.data', delimiter=',')
learning_in = learning[1:-1,:-1]
learning_out = learning[1:-1, -1]
learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)


X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out, test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5, random_state=0)


# Linear SVM
clf_lin = svm.SVC(kernel='linear')
linear_train_error = []
linear_test_error = []
training_samples = []
# Radial SVM
clf_rad = svm.SVC(kernel='rbf')
radial_train_error = []
radial_test_error = []
#polinomial SVM
clf_poly = svm.SVC(kernel='poly', degree=3)
poly_train_error = []
poly_test_error = []

rad_train_sizes, rad_train_scores, rad_valid_scores = learning_curve(clf_rad, X_train, Y_train, cv=5)
print('rad')
lin_train_sizes, lin_train_scores, lin_valid_scores = learning_curve(clf_lin, X_train, Y_train, cv=5)
print('lin')
#poly_train_sizes, poly_train_scores, poly_valid_scores = learning_curve(clf_poly, X_train, Y_train, cv=2)
#print('poly')

stop = timeit.default_timer()
print("time to run:", stop-start)

plt.figure()
plt.title('SVM Learning Curves')
plt.ylim(0, 1)
plt.xlabel("Training examples")
plt.ylabel("Error")

#poly_train_scores = 1 - poly_train_scores
#poly_valid_scores = 1 - poly_valid_scores
lin_train_scores = 1 - lin_train_scores
lin_valid_scores = 1 - lin_valid_scores
rad_train_scores = 1 - rad_train_scores
rad_valid_scores = 1 - rad_valid_scores

lin_train_scores_mean = np.mean(lin_train_scores, axis=1)
lin_train_scores_std = np.std(lin_train_scores, axis=1)
lin_test_scores_mean = np.mean(lin_valid_scores, axis=1)
lin_test_scores_std = np.std(lin_valid_scores, axis=1)
plt.grid()

#plt.fill_between(lin_train_sizes, lin_test_scores_mean - lin_test_scores_std,
                 #lin_test_scores_mean + lin_test_scores_std, alpha=0.1, color="r")
plt.plot(lin_train_sizes, lin_train_scores_mean, '--', color="r",
         label="Linear Training error")
plt.plot(lin_train_sizes, lin_test_scores_mean, 'o-', color="r",
         label="Linear Test error")



rad_train_scores_mean = np.mean(rad_train_scores, axis=1)
rad_train_scores_std = np.std(rad_train_scores, axis=1)
rad_test_scores_mean = np.mean(rad_valid_scores, axis=1)
rad_test_scores_std = np.std(rad_valid_scores, axis=1)
plt.grid()

#plt.fill_between(rad_train_sizes, rad_train_scores_mean - rad_train_scores_std,
                 #rad_train_scores_mean + rad_train_scores_std, alpha=0.1,
                 #color="g")
#plt.fill_between(rad_train_sizes, rad_test_scores_mean - rad_test_scores_std,
                 #rad_test_scores_mean + rad_test_scores_std, alpha=0.1, color="g")
plt.plot(rad_train_sizes, rad_train_scores_mean, '--', color="g", label="Radial Training error")
plt.plot(rad_train_sizes, rad_test_scores_mean, 'o-', color="g", label="Radial Test error")

plt.ylim(0,.5)
plt.legend(loc="best")
plt.show()


radial_train_error_g = []
radial_test_error_g = []
gammas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
for g in gammas:
    clf_rad = svm.SVC(kernel='rbf', gamma=g)
    clf_rad.fit(X_train, Y_train)
    radial_train_error_g.append(1-clf_rad.score(X_train, Y_train))
    radial_test_error_g.append(1-clf_rad.score(X_CV, Y_CV))

plt.figure()
plt.title('SVM Gamma Plot')
plt.ylim(0, .5)
plt.xlabel("Gamma")
plt.ylabel("Error")
plt.plot(gammas, radial_train_error_g, 'o-', color="r", label="Radial Training error")
plt.plot(gammas, radial_test_error_g, 'o-', color="g", label="Radial Test error")
plt.legend(loc="best")

plt.show()

