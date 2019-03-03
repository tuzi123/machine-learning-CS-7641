
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import timeit
from sklearn import cross_validation
import pandas as pd
from scipy import stats

start = timeit.default_timer()
print(start)

learning = np.genfromtxt('DataSet_1/winequality-red.csv', delimiter=';')
learning_in = learning[1:-1,:-1]
learning_out = learning[1:-1, -1]
learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)

X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out, test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.4, random_state=0)

print(X_train.shape)
print(X_CV.shape)
print(X_test.shape)
num_attrs = learning_in.shape[1]
cv_score = []
train_score = []
test_score = []
depth = []


for ind in range(100, X_train.shape[0], 25):
    # slice the data
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train, test_size=ind/X_train.shape[0], random_state=0)
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    train_score.append(1-clf_tree.score(X_train, Y_train))
    cv_score.append(1-clf_tree.score(X_CV, Y_CV))
    test_score.append(1-clf_tree.score(X_test, Y_test))
    depth.append(ind)

stop = timeit.default_timer()

print("time to run:", stop-start)
fig = plt.figure()
plt.title('Decision Tree Training Size ')
plt.xlabel("Training Examples")
plt.ylabel("Error")
plt.ylim(0, .6)
plt.plot(depth, train_score, 'o-', color="r",label="Training error")
plt.plot(depth, cv_score, 'o-', color="g",label="Cross Validation error")
plt.plot(depth, test_score, 'o-', color="b",label="Test error") # removed at the beginning
plt.legend(loc="best")
fig.savefig('Plots/Decision Tree Training Size.png')
plt.close()



best_data_size = (cv_score.index(min(cv_score))+1)*25
print('Best data size: ')
print(best_data_size)

cv_score = []
train_score = []
test_score = []
depth = []


for ind in range(1, 20):
    # slice the data
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train, test_size=best_data_size/X_train.shape[0], random_state=0)
    clf_tree = tree.DecisionTreeClassifier(max_depth=ind)
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    train_score.append(1-clf_tree.score(X_train, Y_train))
    cv_score.append(1-clf_tree.score(X_CV, Y_CV))
    test_score.append(1-clf_tree.score(X_test, Y_test))
    depth.append(ind)

stop = timeit.default_timer()

best_depth = cv_score.index(min(cv_score))+1
print('best depth: ')
print(best_depth)
print("time to run:", stop-start)
fig=plt.figure()
plt.title('Decision Tree Tree Depth ')
plt.xlabel("Tree Depth")
plt.ylabel("Error")
plt.ylim(0, .6)
plt.plot(depth, train_score, 'o-', color="r",label="Training error")
plt.plot(depth, cv_score, 'o-', color="g",label="Cross Validation error")
plt.plot(depth, test_score, 'o-', color="b",label="Test error") # removed
plt.legend(loc="best")
fig.savefig('Plots/Decision Tree Tree Depth.png')
plt.close()
