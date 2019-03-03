import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA, FastICA, RandomizedPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import KFold
from sklearn.metrics import *

from scipy.stats import kurtosis

from timeit import default_timer as time

from data_helper import *
from plot_helper import *

'''
2. Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
'''


class part2():
    def __init__(self):
        self.out_dir = 'output_part2'
        self.save_dir = 'data'
        self.time_filename = './' + self.out_dir + '/time.txt'

    def run(self):
        print('Running part 2')
        with open(self.time_filename, 'w') as text_file:

            t0 = time()
            self.pca_spam()
            text_file.write('pca_spam: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.pca_pima()
            text_file.write('pca_pima: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.ica_spam()
            text_file.write('ica_spam: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.ica_pima()
            text_file.write('ica_pima: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.rp_spam()
            text_file.write('rp_spam: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.rp_pima()
            text_file.write('rp_pima: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.lda_pima()
            text_file.write('lda_pima: %0.3f seconds\n' % (time() - t0))

            t0 = time()
            self.lda_spam()
            text_file.write('lda_spam: %0.3f seconds\n' % (time() - t0))

        ##
        ## Generate files for best
        ##

        self.best_pca_spam()
        self.best_pca_pima()

        self.best_ica_spam()
        self.best_ica_pima()

        self.best_rp_spam()
        self.best_rp_pima()

        self.best_lda_spam()
        self.best_lda_pima()

    def best_pca_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        pca = PCA(n_components=3)
        X_train_transformed = pca.fit_transform(X_train_scl, y_train)
        X_test_transformed = pca.transform(X_test_scl)

        # save
        filename = './' + self.save_dir + '/spam_pca_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_pca_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_pca_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_pca_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_ica_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        ica = FastICA(n_components=X_train_scl.shape[1])
        X_train_transformed = ica.fit_transform(X_train_scl, y_train)
        X_test_transformed = ica.transform(X_test_scl)

        ## top 2
        kurt = kurtosis(X_train_transformed)
        i = kurt.argsort()[::-1]
        X_train_transformed_sorted = X_train_transformed[:, i]
        X_train_transformed = X_train_transformed_sorted[:, 0:2]

        kurt = kurtosis(X_test_transformed)
        i = kurt.argsort()[::-1]
        X_test_transformed_sorted = X_test_transformed[:, i]
        X_test_transformed = X_test_transformed_sorted[:, 0:2]

        # save
        filename = './' + self.save_dir + '/spam_ica_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_ica_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_ica_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_ica_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_rp_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        rp = GaussianRandomProjection(n_components=X_train_scl.shape[1])
        X_train_transformed = rp.fit_transform(X_train_scl, y_train)
        X_test_transformed = rp.transform(X_test_scl)

        ## top 2
        kurt = kurtosis(X_train_transformed)
        i = kurt.argsort()[::-1]
        X_train_transformed_sorted = X_train_transformed[:, i]
        X_train_transformed = X_train_transformed_sorted[:, 0:2]

        kurt = kurtosis(X_test_transformed)
        i = kurt.argsort()[::-1]
        X_test_transformed_sorted = X_test_transformed[:, i]
        X_test_transformed = X_test_transformed_sorted[:, 0:2]

        # save
        filename = './' + self.save_dir + '/spam_rp_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_rp_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_rp_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_rp_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_lda_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        lda = LinearDiscriminantAnalysis(n_components=1)
        X_train_transformed = lda.fit_transform(X_train_scl, y_train)
        X_test_transformed = lda.transform(X_test_scl)

        # save
        filename = './' + self.save_dir + '/spam_lda_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_lda_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_lda_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/spam_lda_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_pca_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        pca = PCA(n_components=1)
        X_train_transformed = pca.fit_transform(X_train_scl, y_train)
        X_test_transformed = pca.transform(X_test_scl)

        # save
        filename = './' + self.save_dir + '/pima_pca_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_pca_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_pca_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_pca_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_ica_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        ica = FastICA(n_components=X_train_scl.shape[1])
        X_train_transformed = ica.fit_transform(X_train_scl, y_train)
        X_test_transformed = ica.transform(X_test_scl)

        ## top 2
        kurt = kurtosis(X_train_transformed)
        i = kurt.argsort()[::-1]
        X_train_transformed_sorted = X_train_transformed[:, i]
        X_train_transformed = X_train_transformed_sorted[:, 0:2]

        kurt = kurtosis(X_test_transformed)
        i = kurt.argsort()[::-1]
        X_test_transformed_sorted = X_test_transformed[:, i]
        X_test_transformed = X_test_transformed_sorted[:, 0:2]

        # save
        filename = './' + self.save_dir + '/pima_ica_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_ica_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_ica_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_ica_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_rp_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        rp = GaussianRandomProjection(n_components=X_train_scl.shape[1])
        X_train_transformed = rp.fit_transform(X_train_scl, y_train)
        X_test_transformed = rp.transform(X_test_scl)

        ## top 2
        kurt = kurtosis(X_train_transformed)
        i = kurt.argsort()[::-1]
        X_train_transformed_sorted = X_train_transformed[:, i]
        X_train_transformed = X_train_transformed_sorted[:, 0:2]

        kurt = kurtosis(X_test_transformed)
        i = kurt.argsort()[::-1]
        X_test_transformed_sorted = X_test_transformed[:, i]
        X_test_transformed = X_test_transformed_sorted[:, 0:2]

        # save
        filename = './' + self.save_dir + '/pima_rp_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_rp_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_rp_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_rp_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def best_lda_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()

        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        lda = LinearDiscriminantAnalysis(n_components=2)
        X_train_transformed = lda.fit_transform(X_train_scl, y_train)
        X_test_transformed = lda.transform(X_test_scl)

        # save
        filename = './' + self.save_dir + '/pima_lda_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_lda_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_lda_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)

        filename = './' + self.save_dir + '/pima_lda_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)

    def pca_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        self.pca_analysis(X_train, X_test, y_train, y_test, 'Spam')

    def ica_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        self.ica_analysis(X_train, X_test, y_train, y_test, 'Spam')

    def rp_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        self.rp_analysis(X_train, X_test, y_train, y_test, 'Spam')

    def lda_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        self.lda_analysis(X_train, X_test, y_train, y_test, 'Spam')

    def pca_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()
        self.pca_analysis(X_train, X_test, y_train, y_test, 'Pima')

    def ica_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()
        self.ica_analysis(X_train, X_test, y_train, y_test, 'Pima')

    def rp_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()
        self.rp_analysis(X_train, X_test, y_train, y_test, 'Pima')

    def lda_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()
        self.lda_analysis(X_train, X_test, y_train, y_test, 'Pima')

    def reconstruction_error_lda(self, X_train_scl, y_train, cls):
        rng = range(1, X_train_scl.shape[1] + 1)

        all_mses = np.ndarray([0, len(rng)])
        for n in range(100):
            mses = []
            for i in rng:
                dr = cls(n_components=i)
                scale = StandardScaler()
                scaledX = scale.fit_transform(X_train_scl, y_train)
                scaledX = dr.fit_transform(scaledX, y_train)
                inverse = np.linalg.pinv(dr.components_)
                reconstruction = scale.inverse_transform(np.dot(scaledX, inverse.T))
                mse = sum(map(np.linalg.norm, reconstruction - X_train_scl))
                mses.append(mse)

            all_mses = np.vstack([all_mses, mses])

        return all_mses, rng

    def reconstruction_error(self, X_train_scl, cls):
        rng = range(1, X_train_scl.shape[1] + 1)

        all_mses = np.ndarray([0, len(rng)])
        for n in range(100):
            mses = []
            for i in rng:
                dr = cls(n_components=i)
                #X_transformed = dr.fit_transform(X_train_scl)
                #X_projected = dr.inverse_transform(X_transformed)
                #mse = mean_squared_error(X_train_scl, X_projected)
                #mses.append(mse)
                # print(i, mse)
                scale = StandardScaler()
                scaledX = scale.fit_transform(X_train_scl)
                scaledX = dr.fit_transform(scaledX)
                inverse = np.linalg.pinv(dr.components_)
                reconstruction = scale.inverse_transform(np.dot(scaledX, inverse.T))
                mse = sum(map(np.linalg.norm, reconstruction - X_train_scl))
                mses.append(mse)
            all_mses = np.vstack([all_mses, mses])

        return all_mses, rng

    def plot_scatter(self, X, y, title, filename, f0_name='feature 1', f1_name='feature 2', x0_i=0, x1_i=1):
        y.shape = (y.shape[0],)

        plt.clf()
        plt.cla()

        for i in np.unique(y):
            plt.scatter(X[y == i, x0_i], X[y == i, x1_i], label=i, alpha=.5)

        plt.title(title)
        plt.xlabel(f0_name)
        plt.ylabel(f1_name)

        plt.legend(loc="best")

        plt.savefig(filename)
        plt.close('all')

    def plot_explained_variance(self, pca, title, filename):
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        rng = np.arange(1, pca.explained_variance_ratio_.shape[0] + 1, 1)

        plt.bar(rng, pca.explained_variance_ratio_,
                alpha=0.5, align='center',
                label='Individual Explained Variance')

        plt.step(rng, np.cumsum(pca.explained_variance_ratio_),
                 where='mid', label='Cumulative Explained Variance')

        plt.legend(loc='best')

        plt.title(title)

        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.savefig(filename)
        plt.close('all')

    def lda_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        ##
        ## Plots
        ##
        ph = plot_helper()

        scores = []
        train_scores = []
        rng = range(1, X_train_scl.shape[1] + 1)
        for i in rng:
            lda = LinearDiscriminantAnalysis(n_components=i)
            cv = KFold(X_train_scl.shape[0], 3, shuffle=True)

            # cross validation
            cv_scores = []
            for (train, test) in cv:
                lda.fit(X_train_scl[train], y_train[train])
                score = lda.score(X_train_scl[test], y_train[test])
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)
            scores.append(mean_score)

            # train score
            lda = LinearDiscriminantAnalysis(n_components=i)
            lda.fit(X_train_scl, y_train)
            train_score = lda.score(X_train_scl, y_train)
            train_scores.append(train_score)

            print(i, mean_score)

        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (LDA) for ' + data_set_name
        name = data_set_name.lower() + '_lda_score'
        filename = './' + self.out_dir + '/' + name + '.png'

        ph.plot_series(rng,
                       [scores, train_scores],
                       [None, None],
                       ['cross validation score', 'training score'],
                       cm.viridis(np.linspace(0, 1, 2)),
                       ['o', '*'],
                       title,
                       'n_components',
                       'Score',
                       filename)
        '''
        ##
        ## Reconstruction Error
        ##
        all_mses, rng = self.reconstruction_error_lda(X_train_scl,  y_train, LinearDiscriminantAnalysis)

        title = 'Reconstruction Error (LDA) for ' + data_set_name
        name = data_set_name.lower() + '_lda_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        ph.plot_series(rng,
                       [all_mses.mean(0)],
                       [all_mses.std(0)],
                       ['mse'],
                       ['red'],
                       ['o'],
                       title,
                       'Number of Features',
                       'Mean Squared Error',
                       filename)
        '''

    def rp_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)

        ks = []
        for i in range(1000):
            ##
            ## Random Projection
            ##
            rp = GaussianRandomProjection(n_components=X_train_scl.shape[1])
            rp.fit(X_train_scl)
            X_train_rp = rp.transform(X_train_scl)

            ks.append(kurtosis(X_train_rp))

        mean_k = np.mean(ks, 0)

        ##
        ## Plots
        ##
        ph = plot_helper()

        title = 'Kurtosis (Randomized Projection) for ' + data_set_name
        name = data_set_name.lower() + '_rp_kurt'
        filename = './' + self.out_dir + '/' + name + '.png'

        ph.plot_simple_bar(np.arange(1, len(mean_k) + 1, 1),
                           mean_k,
                           np.arange(1, len(mean_k) + 1, 1).astype('str'),
                           'Feature Index',
                           'Kurtosis',
                           title,
                           filename)
        ##
        ## Reconstruction Error
        ##
        all_mses, rng = self.reconstruction_error(X_train_scl, GaussianRandomProjection)

        title = 'Reconstruction Error (RP) for ' + data_set_name
        name = data_set_name.lower() + '_rp_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        ph.plot_series(rng,
                       [all_mses.mean(0)],
                       [all_mses.std(0)],
                       ['mse'],
                       ['red'],
                       ['o'],
                       title,
                       'Number of Features',
                       'Reconstruction Error',
                       filename)

    def pca_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        ##
        ## PCA
        ##
        pca = PCA(n_components=X_train_scl.shape[1], svd_solver='full')
        X_pca = pca.fit_transform(X_train_scl)

        ##
        ## Plots
        ##
        ph = plot_helper()

        ##
        ## Explained Variance Plot
        ##
        title = 'Explained Variance (PCA) for ' + data_set_name
        name = data_set_name.lower() + '_pca_evar_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        self.plot_explained_variance(pca, title, filename)

        ##
        ## Reconstruction Error
        ##
        all_mses, rng = self.reconstruction_error(X_train_scl, PCA)

        title = 'Reconstruction Error (PCA) for ' + data_set_name
        name = data_set_name.lower() + '_pca_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        ph.plot_series(rng,
                       [all_mses.mean(0)],
                       [all_mses.std(0)],
                       ['mse'],
                       ['red'],
                       ['o'],
                       title,
                       'Number of Features',
                       'Reconstruction Error',
                       filename)

        ##
        ## Manually compute eigenvalues
        ##
        cov_mat = np.cov(X_train_scl.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
        print(eigen_values)
        sorted_eigen_values = sorted(eigen_values, reverse=True)

        title = 'Eigen Values (PCA) for ' + data_set_name
        name = data_set_name.lower() + '_pca_eigen'
        filename = './' + self.out_dir + '/' + name + '.png'

        ph.plot_simple_bar(np.arange(1, len(sorted_eigen_values) + 1, 1),
                           sorted_eigen_values,
                           np.arange(1, len(sorted_eigen_values) + 1, 1).astype('str'),
                           'Principal Components',
                           'Eigenvalue',
                           title,
                           filename)

        ## TODO Factor this out to new method
        ##
        ## Scatter
        ##
        '''
        pca = PCA(n_components=2, svd_solver='full')
        X_pca = pca.fit_transform(X_train_scl)

        title = 'PCA Scatter: Wine'
        filename = './' + self.out_dir + '/wine_pca_sc.png'
        self.plot_scatter(X_pca, y_train, title, filename, f0_name='feature 1', f1_name='feature 2', x0_i=0, x1_i=1)
        '''

    def ica_analysis(self, X_train, X_test, y_train, y_test, data_set_name):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        ##
        ## ICA
        ##
        ica = FastICA(n_components=X_train_scl.shape[1])
        X_ica = ica.fit_transform(X_train_scl)

        ##
        ## Plots
        ##
        ph = plot_helper()

        kurt = kurtosis(X_ica)
        print(kurt)

        title = 'Kurtosis (FastICA) for ' + data_set_name
        name = data_set_name.lower() + '_ica_kurt'
        filename = './' + self.out_dir + '/' + name + '.png'

        ph.plot_simple_bar(np.arange(1, len(kurt) + 1, 1),
                           kurt,
                           np.arange(1, len(kurt) + 1, 1).astype('str'),
                           'Feature Index',
                           'Kurtosis',
                           title,
                           filename)

        ##
        ## Reconstruction Error
        ##
        all_mses, rng = self.reconstruction_error(X_train_scl, FastICA)

        title = 'Reconstruction Error (ICA) for ' + data_set_name
        name = data_set_name.lower() + '_ica_rec_err'
        filename = './' + self.out_dir + '/' + name + '.png'
        ph.plot_series(rng,
                       [all_mses.mean(0)],
                       [all_mses.std(0)],
                       ['mse'],
                       ['red'],
                       ['o'],
                       title,
                       'Number of Features',
                       'Reconstruction Error',
                       filename)


def main():
    p = part2()
    p.run()


if __name__ == '__main__':
    main()

