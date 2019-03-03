
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
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


num_attrs = learning_in.shape[1]
X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out, test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5, random_state=0)
X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train, test_size=800/X_train.shape[0], random_state=0)

test_score_n_est = []
train_score_n_est = []
cv_score_n_est = []
n_est = []

for i in range(1, 200):
    clf_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),
                         algorithm="SAMME",
                         n_estimators=i)
    # clf_tree = GradientBoostingClassifier(n_estimators=i, random_state=0)
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    test_score_n_est.append(1-clf_tree.score(X_test, Y_test))
    train_score_n_est.append(1-clf_tree.score(X_train, Y_train))
    cv_score_n_est.append(1-clf_tree.score(X_CV, Y_CV))
    n_est.append(i)

stop = timeit.default_timer()
print("time to run:", stop-start)
fig1 = plt.figure()
plt.title('Boosted Decision Tree Number of Estimators')
plt.xlabel("Estimators")
plt.ylabel("Error")
plt.plot(n_est, train_score_n_est, 'o-', color="r", label="Training error")
plt.plot(n_est, cv_score_n_est, 'o-', color="g", label="Cross Validation error")
plt.plot(n_est, test_score_n_est, 'o-', color="b",label="Testing error") # removed
plt.legend(loc="best")
fig1.savefig('Plots/Boosted Decision Tree Number of Estimators.png')
plt.close()


Best_n_est = cv_score_n_est.index(min(cv_score_n_est))+1
print('Best estimator: ',Best_n_est)
# print(min(cv_score_n_est))

test_score_lr = []
train_score_lr = []
cv_score_lr = []
learn_rates = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
for lr in learn_rates:
    clf_tree = AdaBoostClassifier(DecisionTreeClassifier(),
                                  algorithm="SAMME",
                                  n_estimators=Best_n_est,learning_rate=lr)
    # clf_tree = GradientBoostingClassifier(n_estimators=Best_n_est, learning_rate=lr, random_state=0)
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    test_score_lr.append(1-clf_tree.score(X_test, Y_test))
    train_score_lr.append(1-clf_tree.score(X_train, Y_train))
    cv_score_lr.append(1-clf_tree.score(X_CV, Y_CV))

stop = timeit.default_timer()

best_LR = learn_rates[cv_score_lr.index(min(cv_score_lr))]
# print(min(cv_score_lr))
print('Best learning rate: ', best_LR)
print("time to run:", stop-start)
fig2 = plt.figure()
plt.title('Boosted Decision Tree Learning Rate')
plt.xlabel("Learning Rate")
plt.ylabel("Error")
plt.plot(learn_rates, train_score_lr, 'o-', color="r", label="Training error")
plt.plot(learn_rates, cv_score_lr, 'o-', color="g", label="Cross Validation error")
plt.plot(learn_rates, test_score_lr, 'o-', color="b",label="Testing error") # removed
plt.legend(loc="best")
fig2.savefig('Plots/Boosted Decision Tree Learning Rate.png')
plt.close()

cv_score = []
test_score = []
train_score = []
depth = []
for i in range(1, 10):
    clf_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i),
                                  algorithm="SAMME",
                                  n_estimators=Best_n_est, learning_rate=best_LR)
    #clf_tree = GradientBoostingClassifier(n_estimators=Best_n_est, learning_rate=best_LR, max_depth=i, random_state=0)
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    test_score.append(1-clf_tree.score(X_test, Y_test))
    train_score.append(1-clf_tree.score(X_train, Y_train))
    cv_score.append(1-clf_tree.score(X_CV, Y_CV))
    depth.append(i)

stop = timeit.default_timer()
print("time to run:", stop-start)
fig3 = plt.figure()
plt.title('Boosted Decision Tree Tree Depth ')
plt.xlabel("Tree Depth")
plt.ylabel("Error")
plt.plot(depth, train_score, 'o-', color="r", label="Training error")
plt.plot(depth, cv_score, 'o-', color="g", label="Cross Validation error")
plt.plot(depth, test_score, 'o-', color="b", label="Testing error") # removed
plt.legend(loc="best")
fig3.savefig('Plots/Boosted Decision Tree Tree Depth.png')
plt.close()

best_depth = cv_score.index(min(cv_score))+1
# print(min(cv_score))
print('Best depth: ',best_depth)

train_size = []
cv_score1 = []
test_score1 = []
train_score1 = []
for ind in range(100, X_train.shape[0], 100):
    # slice the data
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train, test_size=ind/X_train.shape[0], random_state=0)
    clf_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=best_depth),
                                  algorithm="SAMME",
                                  n_estimators=Best_n_est, learning_rate=best_LR)
    # clf_tree = GradientBoostingClassifier(n_estimators=12, random_state=0)
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    train_score1.append(1-clf_tree.score(X_train, Y_train))
    cv_score1.append(1-clf_tree.score(X_CV, Y_CV))
    test_score1.append(1-clf_tree.score(X_test, Y_test))
    train_size.append(ind)

fig4 = plt.figure()
plt.title('Boosted Decision Tree Error VS Training Size ')
plt.xlabel("Training Examples")
plt.ylabel("Error")
plt.ylim(0, .6)
plt.plot(train_size, train_score1, 'o-', color="r",label="Training error")
plt.plot(train_size, cv_score1, 'o-', color="g",label="Cross Validation error")
plt.plot(train_size, test_score1, 'o-', color="b",label="Test error") # removed at the beginning
plt.legend(loc="best")
fig4.savefig('Plots/Boosted Decision Tree Error VS Training Size.png')
plt.close()