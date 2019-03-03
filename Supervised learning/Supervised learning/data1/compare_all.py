import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import timeit
from sklearn import cross_validation
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn import svm
from sklearn import neighbors
from keras.utils.np_utils import to_categorical
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 7
np.random.seed(seed)
learning = np.genfromtxt('DataSet_1/winequality-red.csv', delimiter=';')
learning_in = learning[1:-1,:-1]
learning_out = learning[1:-1, -1]
learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)

X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out, test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5, random_state=0)

"""
Optimal Hyperparameters for each learning algorithm
"""
#Decision Tree
DT_instances = 638
DT_depth = 8

#Boosted Decision Tree
BDT_n_est = 60
BDT_lr = 0.3
BDT_depth = 8

#Neural Network
ANN_lr = .85
ANN_m = .85
ANN_epochs = 200

#SVM
SVM_kernel = 'rbf'
SVM_gamma = .1

#KNN
KNN_weights = 'distance'
KNN_k = 49

"""
Testing the basic Decision Tree
"""
X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train, test_size=DT_instances/X_train.shape[0], random_state=0)

clf_tree = tree.DecisionTreeClassifier(max_depth=6)
clf_tree = clf_tree.fit(X_train_step, Y_train_step)
DT_train_error = 1-clf_tree.score(X_train, Y_train)
DT_cv_error =1-clf_tree.score(X_CV, Y_CV)
DT_test_error =1-clf_tree.score(X_test, Y_test)

print("The Basic Decision tree had a Training Error of: {0}".format(DT_train_error))
print("The Basic Decision tree had a Cross-Validation Error of: {0}".format(DT_cv_error))
print("The Basic Decision tree had a Testing Error of: {0}".format(DT_test_error))
print('\n')
"""
Testing the boosted Decision Tree,
a maximum depth of 2 nodes, a learning rate of .1, and using 16 estimators
"""
clf_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=BDT_depth),
                                  algorithm="SAMME",
                                  n_estimators=BDT_n_est, learning_rate=BDT_lr)
# clf_tree = GradientBoostingClassifier(n_estimators=BDT_n_est, learning_rate=BDT_lr, max_depth=BDT_depth, random_state=0)
clf_tree = clf_tree.fit(X_train_step, Y_train_step)
BDT_test_error = 1-clf_tree.score(X_test, Y_test)
BDT_train_error = 1-clf_tree.score(X_train, Y_train)
BDT_cv_error = 1-clf_tree.score(X_CV, Y_CV)

print("The Boosted Decision tree had a Training Error of: {0}".format(BDT_train_error))
print("The Boosted Decision tree had a Cross-Validation Error of: {0}".format(BDT_cv_error))
print("The Boosted Decision tree had a Testing Error of: {0}".format(BDT_test_error))
print('\n')


Y_train1 = to_categorical(Y_train)
Y_test1 = to_categorical(Y_test)
Y_CV1 = to_categorical(Y_CV)
model = Sequential()
model.add(Dense(input_dim=learning_in.shape[1], output_dim=learning_in.shape[1]/2, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(9, input_dim=learning_in.shape[1]/2, init='uniform'))
model.add(Activation('sigmoid'))
print('compiling')
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
model.fit(X_train, Y_train1, nb_epoch=ANN_epochs, validation_data=(X_CV, Y_CV1), batch_size=10, verbose=0)
ANN_train_error =1- model.evaluate(X_train, Y_train1, verbose=0)[1]
ANN_cv_error = 1-model.evaluate(X_CV, Y_CV1, verbose=0)[1]
ANN_test_error = 1-model.evaluate(X_test, Y_test1, verbose=0)[1]

print("The ANN had a Training Error of: {0}".format(ANN_train_error))
print("The ANN had a Cross-Validation Error of: {0}".format(ANN_cv_error))
print("The ANN had a Testing Error of: {0}".format(ANN_test_error))
print('\n')

"""
Testing the Support Vector Machine with a radial kernel
"""

clf_rad = svm.SVC(kernel=SVM_kernel, gamma=SVM_gamma)
clf_rad.fit(X_train, Y_train)
SVM_train_error = 1-clf_rad.score(X_train, Y_train)
SVM_cv_error = 1-clf_rad.score(X_CV, Y_CV)
SVM_test_error = 1-clf_rad.score(X_test, Y_test)

print("The SVM had a Training Error of: {0}".format(SVM_train_error))
print("The SVM had a Cross-Validation Error of: {0}".format(SVM_cv_error))
print("The SVM had a Testing Error of: {0}".format(SVM_test_error))
print('\n')

"""
Testing K-Nearest Neighbors with distance based weighing and 49 neighbors being considered
"""
knn_dist = neighbors.KNeighborsClassifier(n_neighbors=KNN_k, weights=KNN_weights)
knn_dist.fit(X_train, Y_train)
KNN_train_error = 1-knn_dist.score(X_train, Y_train)
KNN_cv_error = 1-knn_dist.score(X_CV, Y_CV)
KNN_test_error = 1-knn_dist.score(X_test, Y_test)

print("The KNN had a Training Error of: {0}".format(KNN_train_error))
print("The KNN had a Cross-Validation Error of: {0}".format(KNN_cv_error))
print("The KNN had a Testing Error of: {0}".format(KNN_test_error))
print('\n')

fig = plt.figure()
N = 5
width = .35
space = .35

train_errors = [DT_train_error, BDT_train_error, ANN_train_error, SVM_train_error, KNN_train_error]
cv_errors = [DT_cv_error, BDT_cv_error, ANN_cv_error, SVM_cv_error, KNN_cv_error]
test_errors = [DT_test_error, BDT_test_error, ANN_test_error, SVM_test_error, KNN_test_error]

ind = np.arange(N)*2
labels = ['DT', 'BDT', 'ANN','SVM','KNN']
position = [1,3,5,7,9]
plt.bar(ind, train_errors, width, color='r', label='Training Error')
plt.bar(ind+width, cv_errors, width, color='g', label='Cross Validation Error')
plt.bar(ind+2*width, test_errors, width, color='b', label='Test Error')
plt.xticks(position,labels)
#plt.bar(ind+3*width, 5*[0], width, color='b')
plt.title('Errors for different models')
plt.ylabel("Error")
plt.ylim(0,.8)
plt.legend(loc="best")
fig.savefig('Plots/Errors for different models.png')
plt.close()