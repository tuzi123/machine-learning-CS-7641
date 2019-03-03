
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from scipy import stats
from sklearn import cross_validation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


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


m2_train_error = []
m2_test_error = []
m2_cv_error = []
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_CV = to_categorical(Y_CV)

for e in range(0, 1000, 100):
        model1 = Sequential()
        model1.add(Dense(input_dim=learning_in.shape[1], output_dim=learning_in.shape[1]/2, init='uniform'))
        model1.add(Activation('relu'))
        model1.add(Dense(9, input_dim=learning_in.shape[1]/2, init='uniform'))
        model1.add(Activation('sigmoid'))
        print('compiling')
        model1.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
        model1.fit(X_train, Y_train, nb_epoch=e, validation_data=(X_test, Y_test), batch_size=10)
        epochs.append(e)
        ANN_train_error = 1 - model1.evaluate(X_train, Y_train, verbose=0)[1]
        ANN_cv_error = 1 - model1.evaluate(X_CV, Y_CV, verbose=0)[1]
        ANN_test_error = 1 - model1.evaluate(X_test, Y_test, verbose=0)[1]
        m1_train_error.append(ANN_train_error)
        m1_cv_error.append(ANN_cv_error)
        m1_test_error.append(ANN_test_error)


fig = plt.figure()
plt.title('ANN Learning Curve')
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.plot(epochs,  m1_train_error, 'o-', color='r', label="Training error")
plt.plot(epochs,  m1_cv_error, 'o-', color='g', label="Cross Validation error")
plt.plot(epochs, m1_test_error, 'o-', color='b', label='Testing error')


plt.legend(loc="best")
fig.savefig('Plots/ANN Learning Rate.png')
plt.close()
