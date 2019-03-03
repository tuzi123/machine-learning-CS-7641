
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from scipy import stats
from sklearn import cross_validation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

seed = 7
np.random.seed(seed)

learning = np.genfromtxt('DataSet_1/winequality-red.csv', delimiter=';')
learning_in = learning[1:-1,:-1]
learning_out = learning[1:-1, -1]
learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)

X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out, test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5, random_state=0)

epochs = []
m1_train_error = []
m1_test_error = []
m1_cv_error = []

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_CV = to_categorical(Y_CV)

m1_train_error = []
m1_CV_error = []
m1_momentum = []
m1_learning_rate = []
m1_runtime = []

for lr in [.25, .4, .55, .7, .85]:
    for mome in [.25, .4, .55, .7, .85]:
        model1 = Sequential()
        model1.add(Dense(input_dim=learning_in.shape[1], output_dim=learning_in.shape[1] / 2, init='uniform'))
        model1.add(Activation('relu'))
        model1.add(Dense(9, input_dim=learning_in.shape[1] / 2, init='uniform'))
        model1.add(Activation('sigmoid'))
        print('compiling')
        model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model1.fit(X_train, Y_train, nb_epoch=500, validation_data=(X_CV, Y_CV), batch_size=10)
        m1_momentum.append(mome)
        m1_learning_rate.append(lr)
        ANN_train_error = 1 - model1.evaluate(X_train, Y_train, verbose=0)[1]
        ANN_cv_error = 1 - model1.evaluate(X_CV, Y_CV, verbose=0)[1]
        ANN_test_error = 1 - model1.evaluate(X_test, Y_test, verbose=0)[1]
        m1_train_error.append(ANN_train_error)
        m1_CV_error.append(ANN_cv_error)
        m1_test_error.append(ANN_test_error)


M1_X, M1_Y = np.meshgrid([.25, .4, .55, .7, .85], [.25, .4, .55, .7, .85])
M2_X, M2_Y = np.meshgrid([.25, .4, .55, .7, .85], [.25, .4, .55, .7, .85])
# print(M1_X)
# print(M1_Y)
m1_train_scoring_new = np.asarray(m1_train_error).reshape(M1_X.shape)
m1_test_scoring_new = np.asarray(m1_CV_error).reshape(M1_X.shape)
# print(m1_train_scoring_new)
# print(m1_test_scoring_new)

zmax = m1_train_scoring_new.max()
zmin = m1_train_scoring_new.min()
fig = plt.figure()
plt.title('ANN GridSearch Training')
plt.xlabel("Momentum")
plt.ylabel("Learning Rate")
plt.pcolormesh(M1_X,  M1_Y, m1_train_scoring_new, cmap='RdBu', vmin=zmin, vmax=zmax)
plt.colorbar()
plt.legend(loc="best")
fig.savefig('Plots/ANN GridSearch Train.png')
plt.close()

zmax = m1_test_scoring_new.max()
zmin = m1_test_scoring_new.min()
fig = plt.figure()
plt.title('ANN GridSearch Testing')
plt.xlabel("Momentum")
plt.ylabel("Learning Rate")
plt.pcolormesh(M1_X, M1_Y, m1_test_scoring_new, cmap='RdBu', vmin=zmin, vmax=zmax)
plt.colorbar()
fig.savefig('Plots/ANN GridSearch CV.png')
plt.close()
