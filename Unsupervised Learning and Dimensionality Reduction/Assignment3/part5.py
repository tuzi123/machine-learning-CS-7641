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
from part4 import *

'''
5. Apply the clustering algorithms to the same dataset to which you just applied the dimensionality
reduction algorithms (you've probably already done this), treating the clusters as if they were new
features. In other words, treat the clustering algorithms as if they were dimensionality reduction
algorithms. Again, rerun your neural network learner on the newly projected data.
'''        
class part5():
    def __init__(self):
        self.out_dir = 'output_part5'
        self.part4 = part4()
        self.part4.out_dir = self.out_dir
        self.part4.time_filename = './' + self.out_dir + '/time2.txt'
        self.part4.nn_time_filename = './' + self.out_dir + '/nn_time.txt'

    def run(self):
        print('Running part 5')
    
        filename = './' + self.out_dir + '/time.txt'
        with open(filename, 'w') as text_file:
            
            t0 = time()
            self.nn_pca_cluster_spam()
            text_file.write('nn_pca_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_ica_cluster_spam()
            text_file.write('nn_ica_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_rp_cluster_spam()
            text_file.write('nn_rp_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_lda_cluster_spam()
            text_file.write('nn_lda_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.nn_spam_orig()
            text_file.write('nn_spam_orig: %0.3f seconds\n' % (time() - t0))
            
            
    def nn_spam_orig(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        self.part4.nn_analysis(X_train_scl, X_test_scl, y_train, y_test, 'Spam', 'Neural Network Original')

    def nn_pca_cluster_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_kmeans_pca_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network PCA K-Means')
        
        X_train, X_test, y_train, y_test = dh.get_spam_data_gmm_pca_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network PCA GMM')
        
    def nn_ica_cluster_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_kmeans_ica_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network ICA K-Means')
        
        X_train, X_test, y_train, y_test = dh.get_spam_data_gmm_ica_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network ICA GMM')
        
    def nn_rp_cluster_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_kmeans_rp_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network RP K-Means')
        
        X_train, X_test, y_train, y_test = dh.get_spam_data_gmm_rp_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network RP GMM')
        
    def nn_lda_cluster_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_kmeans_lda_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network LDA K-Means')
    
        X_train, X_test, y_train, y_test = dh.get_spam_data_gmm_lda_best()
        self.part4.nn_analysis(X_train, X_test, y_train, y_test, 'Spam', 'Neural Network LDA GMM')
        
def main():    
    p = part5()
    p.run()

if __name__== '__main__':
    main()
    
